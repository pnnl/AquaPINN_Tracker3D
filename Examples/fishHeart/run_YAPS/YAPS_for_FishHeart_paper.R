# =============================================================================
# YAPS_for_FishHeart_paper.R
#
# Runs YAPS positioning for the FishHeart deployment using a pre-built
# decodes4YAPS.csv (already time-synced detections).
#
# Input files (relative to this script):
#     ../config/YAPS/receiver_locs_w_beacons_meters_numeric_tag.csv
#     ../config/YAPS/GPS_Track.csv
#     ../config/YAPS/Fishheart_EA_test_description.csv
#     ../config/YAPS/decodes4YAPS.csv
#
# Output:
#     results/tag_*.png / tag_*_boxplot.png / tag_*.csv
# =============================================================================

rm(list = ls())
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(sp)
  library(leaflet)
  library(lubridate)
  library(sf)
  library(yaps)
})

set.seed(42)
options(digits.secs = 6)
options(digits = 15)

# -----------------------------------------------------------------------------
# 0. Paths  (all relative to this script's directory)
# -----------------------------------------------------------------------------
script_dir <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  fa <- args[grep("^--file=", args)]
  if (length(fa) > 0L) {
    normalizePath(dirname(sub("^--file=", "", fa[1])),
                  winslash = "/", mustWork = FALSE)
  } else if (!is.null(sys.frames()) && length(sys.frames()) > 0L &&
             !is.null(sys.frame(1)$ofile)) {
    normalizePath(dirname(sys.frame(1)$ofile),
                  winslash = "/", mustWork = FALSE)
  } else {
    getwd()
  }
}, error = function(e) getwd())
if (dir.exists(script_dir)) setwd(script_dir)
cat("Working directory: ", getwd(), "\n", sep = "")

CONFIG_DIR   <- "../config/YAPS"
HYDROS_FILE  <- file.path(CONFIG_DIR, "receiver_locs_w_beacons_meters_numeric_tag.csv")
GPS_FILE     <- file.path(CONFIG_DIR, "GPS_Track.csv")
EA_FILE      <- file.path(CONFIG_DIR, "Fishheart_EA_test_description.csv")
DECODES_FILE <- file.path(CONFIG_DIR, "decodes4YAPS.csv")

OUT_DIR <- "results"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 1. Hydrophone config  (already in metres)
# -----------------------------------------------------------------------------
hydros <- as.data.table(read.csv(HYDROS_FILE, stringsAsFactors = FALSE))
hydros$sync_tag[hydros$sync_tag == ""] <- NA
hydros$sync_tag <- as.numeric(hydros$sync_tag)
hydros$serial   <- as.numeric(hydros$serial)
hydros$idx      <- as.numeric(hydros$idx)

# -----------------------------------------------------------------------------
# 2. Pre-synced detections
# -----------------------------------------------------------------------------
detections <- as.data.table(read.csv(DECODES_FILE))
setnames(detections, old = "time",  new = "ts")
setnames(detections, old = "rxSN",  new = "serial")
setnames(detections, old = "tagID", new = "tag")
setnames(detections, old = "rxID",  new = "hydro_idx")

detections$ts <- as.POSIXct((detections$ts - 719529) * 86400,
                             origin = "1970-01-01", tz = "UTC")
detections$serial    <- as.numeric(detections$serial)
detections$tag       <- as.numeric(detections$tag)
detections$hydro_idx <- as.numeric(detections$hydro_idx)

detections[, epo  := as.integer(ts)]
detections[, frac := (as.numeric(ts) - floor(as.numeric(ts))) * 1e6]
detections$epo     <- as.numeric(detections$epo)
detections$frac    <- as.numeric(detections$frac) / 1e6
detections$epofrac <- detections$epo + detections$frac
detections$eposync <- detections$epofrac

# -----------------------------------------------------------------------------
# 3. GPS reference track
# -----------------------------------------------------------------------------
gps <- as.data.table(read.csv(GPS_FILE))
gps$ts <- as.POSIXct(as.character(gps$ts),
                     format = "%m/%d/%Y %I:%M:%S", tz = "UTC") + (12 * 60 * 60)
gps <- gps[ts %between% as.POSIXct(c('2024-03-21 12:06:00',
                                      '2024-03-21 12:46:00'), tz = "UTC")]

# -----------------------------------------------------------------------------
# 4. Build YAPS hydro frame
# -----------------------------------------------------------------------------
detections_synced <- detections

hydros_yaps <- copy(hydros)
setnames(hydros_yaps, old = "x", new = "hx")
setnames(hydros_yaps, old = "y", new = "hy")
setnames(hydros_yaps, old = "z", new = "hz")

# -----------------------------------------------------------------------------
# 5. Tag table
# -----------------------------------------------------------------------------
focal_tags_code <- c('G72256C58', 'G7254813E', 'G72285C6F', 'G726777EF')
focal_tags      <- c(0L, 1L, 2L, 3L)

# -----------------------------------------------------------------------------
# 6. Positioning loop
# -----------------------------------------------------------------------------
track_solved_all <- list()

for (tag_idx in seq_along(focal_tags)) {
  focal_tag      <- focal_tags[tag_idx]
  focal_tag_code <- focal_tags_code[tag_idx]

  synced_dat_tag <- detections_synced[detections_synced$tag == focal_tag, ]

  if (tag_idx <= 3L) {
    PRI <- 1; rbi_min <- 0.9; rbi_max <- 1.1
  } else {
    PRI <- 2; rbi_min <- 1.9; rbi_max <- 2.1
  }

  synced_dat <- synced_dat_tag
  gps_track  <- gps

  toa <- getToaYaps(synced_dat = synced_dat, hydros_yaps,
                    rbi_min, rbi_max, pingType = "sbi")

  inp <- getInp(hydros_yaps, toa, E_dist = "Mixture", n_ss = 2,
                pingType = "sbi", sdInits = 1,
                rbi_min = rbi_min, rbi_max = rbi_max,
                ss_data_what = "data", ss_data = rep(1500, nrow(toa)),
                z_vec = rep(10, nrow(toa)))

  yaps_out <- runYaps(inp, maxIter = 10000,
                      getPlsd = TRUE, getRep = TRUE, silent = TRUE)

  track_solved <- list(
    top = yaps_out$track$top,
    x   = yaps_out$track$x,
    y   = yaps_out$track$y,
    z   = yaps_out$track$z
  )
  yaps_out$track$top <- as.POSIXct(yaps_out$track$top, tz = "UTC")

  all_x   <- c(hydros$x, track_solved$x, gps_track$utm_x)
  all_y   <- c(hydros$y, track_solved$y, gps_track$utm_y)
  x_range <- range(all_x, na.rm = TRUE)
  y_range <- range(all_y, na.rm = TRUE)
  buffer  <- 0.05
  xlims   <- c(x_range[1] - diff(x_range)*buffer/2,
               x_range[2] + diff(x_range)*buffer/2)
  ylims   <- c(y_range[1] - diff(y_range)*buffer,
               y_range[2] + diff(y_range)*buffer)

  gps_interped <- sapply(c("utm_x", "utm_y"), function(coord) {
    approx(x = gps_track$ts, y = gps_track[[coord]],
           xout = track_solved$top, rule = 2)$y
  })

  errors <- list()
  for (i in seq_along(track_solved$top)) {
    errors$x[i] <- track_solved$x[i] - gps_interped[i, 1]
    errors$y[i] <- track_solved$y[i] - gps_interped[i, 2]
  }
  errors$median_error <- c(median(abs(errors$x)), median(abs(errors$y)))
  errors$mean_error   <- c(mean(abs(errors$x)),   mean(abs(errors$y)))
  errors$RMSE         <- c(sqrt(sum(abs(errors$x)^2) / length(errors$x)),
                           sqrt(sum(abs(errors$y)^2) / length(errors$y)))

  track_duration <- as.numeric(
    track_solved$top[length(track_solved$top)] - track_solved$top[1],
    units = "secs")
  efficiency <- round(PRI * length(track_solved$top) / track_duration * 100)

  png(sprintf("%s/tag_%d.png", OUT_DIR, tag_idx - 1),
      width = 5000, height = 3000, res = 500)
  layout(matrix(c(1,1,2,3, 1,1,4,5), nrow = 2, byrow = TRUE, ncol = 4))
  plot(y ~ x, data = hydros, asp = 1, xlim = xlims, ylim = ylims,
       xlab = "X-axis", ylab = "Y-axis",
       main = paste("GPS and Solved Track (YAPS) - Efficiency:", efficiency, "%"))
  text(y ~ x, data = hydros, label = idx,      pos = 2)
  text(y ~ x, data = hydros, label = serial,   pos = 3)
  text(y ~ x, data = hydros[!is.na(sync_tag)], label = sync_tag, pos = 1)
  points(y ~ x, data = hydros[!is.na(sync_tag)], pch = 20, col = "skyblue")
  lines(track_solved$y ~ track_solved$x, type = "o",
        pch = 8, lty = 2, col = "red", lwd = 0.5, cex = 0.2)
  lines(utm_y ~ utm_x, data = gps_track, lty = 2)
  legend("topright", legend = c("Track Solved", "GPS Track"),
         col = c("red", "blue"), lty = 2, pch = c(20, NA), lwd = 2)
  plot(track_solved$x ~ track_solved$top, ylab = "X-axis Track", xlab = "Time",
       type = "o", pch = 8, lty = 2, col = "red", lwd = 0.5, cex = 0.2, xaxt = 'n')
  axis.POSIXct(1, at = seq(min(track_solved$top), max(track_solved$top), by = "mins"),
               format = "%H:%M")
  lines(gps_interped[, 1] ~ track_solved$top, lty = 2, col = "blue")
  plot(track_solved$y ~ track_solved$top, ylab = "Y-axis Track", xlab = "Time",
       type = "o", pch = 8, lty = 2, col = "red", lwd = 0.5, cex = 0.2, xaxt = 'n')
  axis.POSIXct(1, at = seq(min(track_solved$top), max(track_solved$top), by = "mins"),
               format = "%H:%M")
  lines(gps_interped[, 2] ~ track_solved$top, lty = 2, col = "blue")
  plot(abs(errors$x) ~ track_solved$top, type = "o", ylab = "X-axis Error",
       xlab = "Time", col = "black", pch = 20, lwd = 1, lty = 1, xaxt = 'n',
       main = paste("Median:", round(errors$median_error[1], 2),
                    "; RMS:", round(errors$RMSE[1], 2)))
  axis.POSIXct(1, at = seq(min(track_solved$top), max(track_solved$top), by = "mins"),
               format = "%H:%M")
  plot(abs(errors$y) ~ track_solved$top, type = "o", ylab = "Y-axis Error",
       xlab = "Time", col = "black", pch = 20, lwd = 1, lty = 1, xaxt = 'n',
       main = paste("Median:", round(errors$median_error[2], 2),
                    "; RMS:", round(errors$RMSE[2], 2)))
  axis.POSIXct(1, at = seq(min(track_solved$top), max(track_solved$top), by = "mins"),
               format = "%H:%M")
  dev.off()

  png(sprintf("%s/tag_%d_boxplot.png", OUT_DIR, tag_idx - 1),
      width = 1000, height = 1500, res = 300)
  boxplot(list(x = abs(errors$x), y = abs(errors$y)),
          main = "Error Boxplot", ylab = "Error (meter)", xlab = "Direction",
          names = c("X-axis", "Y-axis"), col = c("tomato", "skyblue"))
  dev.off()

  track_solved_csv <- data.frame(
    datetime = track_solved$top,
    X        = track_solved$x,
    Y        = track_solved$y,
    "X error" = errors$x,
    "Y error" = errors$y,
    tag_code = rep(focal_tag_code, length(track_solved$top)),
    check.names = FALSE
  )
  write.csv(track_solved_csv,
            sprintf("%s/tag_%d_%s.csv", OUT_DIR, tag_idx - 1, focal_tag_code),
            row.names = FALSE)
  track_solved_all <- rbind(track_solved_all, track_solved_csv)
}

cat(sprintf("\nDone. Results in %s/\n", OUT_DIR))
