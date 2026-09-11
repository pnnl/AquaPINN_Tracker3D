# =============================================================================
# YAPS_for_Sequim_paper.R
#
# Runs YAPS positioning for the Sequim deployment using a pre-built
# decodes4YAPS.csv (already time-synced detections).
#
# Input files (relative to this script):
#     ../config/sequim_hydro_for_yaps.csv          Hydrophone array (ft -> m)
#     ../config/GPS_track.csv                      DGPS reference track (m)
#     ../config/EA_test_description_forPaper.csv   EA test windows
#     ../config/decodes4YAPS.csv                   Pre-synced detections
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
HYDROS_FILE  <- file.path(CONFIG_DIR, "sequim_hydro_for_yaps.csv")
GPS_FILE     <- file.path(CONFIG_DIR, "GPS_track.csv")
EA_FILE      <- file.path(CONFIG_DIR, "EA_test_description_forPaper.csv")
DECODES_FILE <- file.path(CONFIG_DIR, "decodes4YAPS.csv")

OUT_DIR <- "results"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 1. Hydrophone config (feet -> metres)
# -----------------------------------------------------------------------------
hydros <- as.data.table(read.csv(HYDROS_FILE, stringsAsFactors = FALSE))
hydros$sync_tag[hydros$sync_tag == ""] <- NA
hydros$sync_tag <- as.numeric(hydros$sync_tag)
hydros$serial   <- as.numeric(hydros$serial)
hydros$idx      <- as.numeric(hydros$idx)

ft_to_m <- 0.3048
hydros[, `:=`(x = x * ft_to_m, y = y * ft_to_m, z = z * ft_to_m)]

ref <- hydros[idx == 1L]

# -----------------------------------------------------------------------------
# 2. Pre-synced detections
# -----------------------------------------------------------------------------
detections <- as.data.table(read.csv(DECODES_FILE))
detections$ts        <- as.POSIXct(detections$ts,        tz = "UTC")
detections$serial    <- as.numeric(detections$serial)
detections$tag       <- as.numeric(detections$tag)
detections$hydro_idx <- as.numeric(detections$hydro_idx)
detections$epo       <- as.numeric(detections$epo)
detections[, ts_epo  := as.integer(ts)]
detections$epo       <- detections$ts_epo
detections$frac      <- as.numeric(detections$frac)
detections$epofrac   <- detections$epo + detections$frac
detections$eposync   <- detections$epofrac
detections <- detections[ts %between%
  as.POSIXct(c('2024-02-21 11:40:00', '2024-02-21 12:20:00'), tz = "UTC")]

# -----------------------------------------------------------------------------
# 3. GPS reference track
# -----------------------------------------------------------------------------
gps <- as.data.table(read.csv(GPS_FILE))
if ("X" %in% names(gps)) gps[, X := NULL]
gps$utm_y <- gps$y - ref$y
gps$utm_x <- gps$x - ref$x
gps[, ts := as.POSIXct(trimws(gsub("\\s+", " ", as.character(ts))),
                       format = "%m/%d/%Y %H:%M:%S", tz = "UTC")]
setorder(gps, ts)

# -----------------------------------------------------------------------------
# 4. EA test period definitions
# -----------------------------------------------------------------------------
ea_tests <- as.data.table(read.csv(EA_FILE))
ea_tests[, StartTime := as.POSIXct(trimws(StartTime),
                                   format = "%m/%d/%Y %H:%M:%S", tz = "UTC")]
ea_tests[, EndTime   := as.POSIXct(trimws(EndTime),
                                   format = "%m/%d/%Y %H:%M:%S", tz = "UTC")]

# -----------------------------------------------------------------------------
# 5. Build YAPS hydro frames
# -----------------------------------------------------------------------------
hydros_yaps <- copy(hydros)
setnames(hydros_yaps, old = "x", new = "hx")
setnames(hydros_yaps, old = "y", new = "hy")
setnames(hydros_yaps, old = "z", new = "hz")
hydros_yaps[, `:=`(hx = hx - ref$x, hy = hy - ref$y, hz = hz - ref$z)]

hydros[, `:=`(x = x - ref$x, y = y - ref$y, z = z - ref$z)]

stopifnot("idx" %in% names(hydros), "x" %in% names(hydros), "y" %in% names(hydros))
ref <- hydros[as.integer(idx) == 1L][1]

gps[, `:=`(hx = utm_x - ref$x, hy = utm_y - ref$y)]

detections_synced <- detections

# -----------------------------------------------------------------------------
# 6. Tag table
# -----------------------------------------------------------------------------
focal_tags      <- c(0,           1,           2,           3,           4,
                     5,           6,           7,           8)
focal_tags_code <- c('G72285C6F', 'G72D1B46F', 'G7270F55D', 'G7254813E', 'G724601CF',
                     'G726D7656', 'G726777EF', 'G72679FC4', 'G720EA94F')

# -----------------------------------------------------------------------------
# 7. Positioning loop
# -----------------------------------------------------------------------------
track_solved_all <- list()

for (tag_idx in seq_along(focal_tags)) {
  focal_tag      <- focal_tags[tag_idx]
  focal_tag_code <- focal_tags_code[tag_idx]

  synced_dat_tag <- detections_synced[detections_synced$tag == focal_tag, ]

  if (nrow(synced_dat_tag) == 0L) {
    message(sprintf("Skipping tag %d (%s): no detections in the selected time window.",
                    focal_tag, focal_tag_code))
    next
  }

  if (focal_tag %in% c(0, 1)) {
    PRI <- 1; rbi_min <- 0.9;  rbi_max <- 1.1
  } else if (focal_tag %in% c(2, 5, 7)) {
    PRI <- 3; rbi_min <- 2.9;  rbi_max <- 3.1
  } else if (focal_tag %in% c(3, 4)) {
    PRI <- 1; rbi_min <- 1;    rbi_max <- 1
  } else if (focal_tag %in% c(8)) {
    PRI <- 1; rbi_min <- 0.98; rbi_max <- 1.02
  } else if (focal_tag == 6) {
    PRI <- 1.85; rbi_min <- 1.7; rbi_max <- 2.3
  } else {
    stop(sprintf("Unhandled focal_tag: %s", focal_tag))
  }

  synced_dat <- synced_dat_tag
  gps_track  <- gps

  toa <- getToaYaps(synced_dat = synced_dat, hydros_yaps,
                    rbi_min, rbi_max, pingType = "sbi")

  inp <- getInp(hydros_yaps, toa, E_dist = "Mixture", n_ss = 2,
                pingType = "sbi", sdInits = 1,
                rbi_min = rbi_min, rbi_max = rbi_max,
                ss_data_what = "data", ss_data = rep(1485, nrow(toa)))

  yaps_out <- runYaps(inp, maxIter = 10000,
                      getPlsd = TRUE, getRep = TRUE, silent = TRUE)

  track_solved <- list(
    top = yaps_out$track$top,
    x   = yaps_out$track$x,
    y   = yaps_out$track$y,
    z   = yaps_out$track$z
  )
  yaps_out$track$top <- as.POSIXct(yaps_out$track$top, tz = "UTC")

  all_x    <- c(hydros$x, track_solved$x, gps_track$utm_x)
  all_y    <- c(hydros$y, track_solved$y, gps_track$utm_y)
  x_range  <- range(all_x, na.rm = TRUE)
  y_range  <- range(all_y, na.rm = TRUE)
  buffer   <- 0.05
  xlims    <- c(x_range[1] - diff(x_range)*buffer/2,
                x_range[2] + diff(x_range)*buffer/2)
  ylims    <- c(y_range[1] - diff(y_range)*buffer,
                y_range[2] + diff(y_range)*buffer)

  gps_interped <- sapply(c("utm_x", "utm_y"), function(coord) {
    approx(x = gps_track$ts, y = gps_track[[coord]],
           xout = track_solved$top, rule = 2)$y
  })

  period_labels <- vapply(track_solved$top, function(t) {
    idx <- which(t >= ea_tests$StartTime & t <= ea_tests$EndTime)
    if (length(idx) == 0L) NA_character_ else ea_tests$Name[idx[1L]]
  }, character(1))
  in_test <- !is.na(period_labels)

  errors <- list(
    x = track_solved$x - gps_interped[, 1],
    y = track_solved$y - gps_interped[, 2]
  )
  errors$median_error <- c(median(abs(errors$x[in_test])),
                           median(abs(errors$y[in_test])))
  errors$mean_error   <- c(mean(abs(errors$x[in_test])),
                           mean(abs(errors$y[in_test])))
  errors$RMSE         <- c(sqrt(mean(errors$x[in_test]^2)),
                           sqrt(mean(errors$y[in_test]^2)))

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
  for (coord_idx in 1:2) {
    coord_name <- c("X-axis Track", "Y-axis Track")[coord_idx]
    coord_vals <- list(track_solved$x, track_solved$y)[[coord_idx]]
    plot(coord_vals ~ track_solved$top, ylab = coord_name, xlab = "Time",
         type = "o", pch = 8, lty = 2, col = "red", lwd = 0.5, cex = 0.2,
         xaxt = 'n')
    ticks <- seq(min(track_solved$top), max(track_solved$top), by = "mins")
    axis(1, at = ticks, labels = format(ticks, "%H:%M", tz = "UTC"))
    lines(gps_interped[, coord_idx] ~ track_solved$top, lty = 2, col = "blue")
  }
  for (coord_idx in 1:2) {
    err_vals  <- list(errors$x, errors$y)[[coord_idx]]
    axis_name <- c("X-axis Error", "Y-axis Error")[coord_idx]
    plot(abs(err_vals) ~ track_solved$top, type = "o", ylab = axis_name,
         xlab = "Time", col = "black", pch = 20, lwd = 1, lty = 1, xaxt = 'n',
         main = paste("Median:", round(errors$median_error[coord_idx], 2),
                      "; RMS:", round(errors$RMSE[coord_idx], 2)))
    ticks <- seq(min(track_solved$top), max(track_solved$top), by = "mins")
    axis(1, at = ticks, labels = format(ticks, "%H:%M", tz = "UTC"))
  }
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
    period   = period_labels,
    tag_code = rep(focal_tag_code, length(track_solved$top)),
    check.names = FALSE
  )
  write.csv(track_solved_csv,
            sprintf("%s/tag_%d_%s.csv", OUT_DIR, tag_idx - 1, focal_tag_code),
            row.names = FALSE)
  track_solved_all <- rbind(track_solved_all, track_solved_csv)
}

cat(sprintf("\nDone. Results in %s/\n", OUT_DIR))
