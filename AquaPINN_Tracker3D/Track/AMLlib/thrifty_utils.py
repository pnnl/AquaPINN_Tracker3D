from settings import parse_kvconfig
import numpy as np
import glob
import toads_data
import os
import itertools
import collections
import stat_tools
import bisect
from six import string_types
import csv
from scipy.optimize import least_squares
import time


def load_cfg_positions(cfg_file_path):
    d = {}
    with open(cfg_file_path) as f:
        for line in f:
            if line == "\n":
                break
            elif line.split()[0][1].isdigit():
                key = line.split()[0][0:-1]
            else:
                key = line.split()[0][0]
            d[int(key)] = [float(val) for val in line.split()[1:]] 
    return d

#############################################################################################

def load_freqmap(cfg_file_path):

    with open(cfg_file_path) as file_:
        if file_ is None:
            return None
        strings = parse_kvconfig(file_)

        tx_ranges = {}
        rx_offset = {}

        for key, value in strings.items():
            if key[0] == '@':
                rx_offset[int(key[1:])] = float(value)
            else:
                # TODO: use regex
                start, stop = [float(x.strip()) for x in value.split('-')]
                tx_ranges[int(key)] = (start, stop)
                # TODO: ensure that ranges do not overlap

        freq_map = {}
        for rxid, offset in rx_offset.items():
            freq_map[rxid] = {}
            for txid, range_ in tx_ranges.items():
                start, stop = range_
                freq_map[rxid][txid] = (start+offset, stop+offset)
                # print(rxid, txid, freq_map[rxid][txid])

    return freq_map

#############################################################################################

def classify_transmitters(detections, freqmap):
    """Identify transmitter IDs based on the closest nominal frequency."""
    txids = []
    for detection in detections:
        freq = detection.carrier_info.bin + detection.carrier_info.offset
        this_txid = -1  # unidentifier (FIXME: don't use magic number)
        for txid, range_ in freqmap[detection.rxid].items():
            start, stop = range_
            if freq >= start and freq <= stop:
                this_txid = txid
        txids.append(this_txid)
    return txids

#############################################################################################

def identify_transmitters(detections, freqmap):
    """
    Identify transmitters and add TX info to detections.
    The DetectionResult object (detections) is changed in-place.
    """

    # if freqmap is None:
    #     txids = auto_classify_transmitters(detections)
    # else:
    #     txids = classify_transmitters(detections, freqmap)

    txids = classify_transmitters(detections, freqmap)

    for i, detection in enumerate(detections):
        detection.txid = txids[i]
        detections[i] = detection
    return detections

#############################################################################################

def load_toad_files(toad_globs):
    filenames = []
    for toad_glob in toad_globs:
        filenames.extend(glob.glob(toad_glob))

    detections = []
    for filename in filenames:
        with open(filename, 'r') as file_:
            detections.extend(toads_data.load_toad(file_))

    return detections, filenames

#############################################################################################

def identify_duplicates(detections):
    """
    Returns a mask for filtering duplicate detections for the same transmitter.

    The block prior to or after the full detection may contain a portion of
    positioning signal and also trigger a detection. It is thus necessary
    to remove those "duplicate" detections.
    It is assumed that all detections were captured by the same receiver.

    The mask will exclude unidentified detections.
    """
    array = toads_data.toads_array(detections, with_ids=True)

    # Sort by receiver ID, then transmitter ID, then block ID, then timestamp
    idx = np.argsort(array[['rxid', 'txid', 'block', 'timestamp']])

    cur = array[idx]
    prev = np.roll(cur, 1)
    next_ = np.roll(cur, -1)

    # TODO: only filter if SOA is within code_len
    mask_unidentified = (cur['txid'] == -1)  # FIXME: magic number
    mask_prev = ((cur['block'] == prev['block'] + 1) &
                 (cur['energy'] < prev['energy']))
    mask_next = ((cur['block'] == next_['block'] - 1) &
                 (cur['energy'] < next_['energy']))
    mask = ~(mask_prev | mask_next | mask_unidentified)
    reverse_idx = np.argsort(idx)

    return mask[reverse_idx]

#############################################################################################

def filter_duplicates(detections):
    """Return detections with duplicates and unidentified detections removed,
    sorted by timestamp."""
    mask = identify_duplicates(detections)
    filtered = list(itertools.compress(detections, mask))
    filtered.sort(key=lambda x: x.timestamp)
    return filtered

#############################################################################################

def integrate(detections, freqmap=None):
    """Identify and filter."""
    identify_transmitters(detections, freqmap)
    filtered = filter_duplicates(detections)
    return filtered

#############################################################################################

def generate_toads(toad_globs, freqmap):
    detections, filenames = load_toad_files(toad_globs)
    filtered = integrate(detections, freqmap)

    print("Removed {} duplicates / unidentified transmisisons "
          "from {} detections.".format(len(detections)-len(filtered),
                                       len(detections)))
    detections.sort(key=lambda x: x.timestamp)
    return detections


def generate_toads_w_output(output, toad_globs, freqmap):
    detections, filenames = load_toad_files(toad_globs)
    # output.write("# source_files: [%s]\n" % (' '.join(filenames)))
    filtered = integrate(detections, freqmap)

    print("Removed {} duplicates / unidentified transmisisons "
          "from {} detections.".format(len(detections)-len(filtered),
                                       len(detections)))

    for detection in filtered:
        output.write(detection.serialize() + '\n')

#############################################################################################


def generate_CSV_for_TOAs_by_RX_TX(TOAs_by_RX_TX):

    for RX_TX_pair in TOAs_by_RX_TX.keys():

        if not os.path.exists('./2-RXs/RX'+ str(RX_TX_pair[0]) +'/'):
            os.makedirs('./2-RXs/RX'+ str(RX_TX_pair[0]) +'/')
        with open( './2-RXs/RX'+ str(RX_TX_pair[0]) +'/RX_'+str(RX_TX_pair[0])+'_TX_'+str(RX_TX_pair[1])+'_TOA.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['timestamp'] + ['RX_ID'] + ['TX_ID'] + ['TOA'] + ['SNR'])
            for toa_data in TOAs_by_RX_TX[RX_TX_pair]:
                csvwriter.writerow(toa_data)

#############################################################################################

def match_toads(toads, window, min_match=2):
    """Match detections from multiple receivers.

    The coarse timestamp and transmitter ID is used to determine which
    detections originate from the same transmissions.

    Parameters
    ----------
    toads : list of DetectionResult
        List should be sorted by timestamp.
    window : float
        Size of timestamp window in seconds.
    min_match : int
        Minimum number of receivers that should receive a transmission for it
        to be considered a valid match.

    Returns
    -------
    matches : list
        List of matches, each match being a list of indices.
    misses : list
        List of detection indices that could not be matched.
    """
    # pylint: disable=too-many-locals
    num_det = len(toads)

    killed = [False] * len(toads)
    matches = []
    misses = []
    collisions = []

    for i in range(num_det):
        if killed[i]:
            continue

        rx_match = {}
        rx_match[toads[i].rxid] = i

        for j in range(i + 1, num_det):
            if toads[j].txid != toads[i].txid:
                continue
            if toads[j].timestamp > toads[i].timestamp + window:
                break
            killed[j] = True

            if toads[j].rxid in rx_match != -1:
                prev = rx_match[toads[j].rxid]
                collisions.append((prev, j))
                prev_ampl = toads[prev].corr_info.energy
                this_ampl = toads[j].corr_info.energy
                k = prev if prev_ampl > this_ampl else j # select signals with larger amplitude
            else:
                k = j

            rx_match[toads[j].rxid] = k

        
        match = rx_match.values()
        # if len(match) >= min_match:
        #     print(len(match))
        if len(match) >= min_match and toads[i].txid != -1:
            matches.append(match)
        else:
            misses.append(i)

    return matches, misses, collisions


#############################################################################################


def load_matches(file_):
    """Load match data from .match file."""
    matches = []
    if isinstance(file_, str):
        file_ = open(file_, 'r')
    for line in file_:
        if len(line) == 0 or line[0] == '#':
            continue
        match = map(int, line.split())
        matches.append(match)
    return matches


#############################################################################################

def save_matches(matches, file_):
    """Save match data to .match file."""
    for match in matches:
        file_.write(' '.join(map(str, match)) + '\n')


#############################################################################################

def extract_match_matrix(detections, matches, rxids, txids=None):
    matrix = []
    for match in matches:
        match_rxids = [detections[m].rxid for m in match]
        row = [None] * len(rxids)
        for i, rxid in enumerate(rxids):
            if rxid not in match_rxids:
                break
            if txids is not None and detections[match[0]].txid not in txids:
                break
            assert row[i] is None
            row[i] = match[match_rxids.index(rxid)]
        else:
            matrix.append(row)
    return matrix



#############################################################################################


SPEED_OF_LIGHT = 2.997e8
MAX_TDOA = 30e3 / SPEED_OF_LIGHT


TdoaInfo = collections.namedtuple('TdoaInfo', [
    'rx0', 'rx1', 'tdoa', 'snr', 'model_quality', 'det0_idx', 'det1_idx'])

TdoaGroup = collections.namedtuple('TdoaGroup', [
    'group_id', 'timestamp', 'tx', 'tdoas'])

TDOA_DTYPE = {'names': ('rx0', 'rx1', 'tdoa', 'snr', 'model_quality',
                        'det0_idx', 'det1_idx'),
              'formats': ('i4', 'i4', 'f8', 'f8', 'f8', 'i4', 'i4')}

MATRIX_DTYPE = {'names': ('group_id', 'timestamp', 'tx') + TDOA_DTYPE['names'],
                'formats': ('i4', 'f8', 'i4') + TDOA_DTYPE['formats']}


def make_detection_extractor(detections, matches):
    rxpair_detections = collections.defaultdict(list)
    for group in matches:
        for det0_id, det1_id in itertools.combinations(group, 2):
            det0 = detections[det0_id]
            det1 = detections[det1_id]
            if det0.rxid > det1.rxid:
                det0, det1 = det1, det0
            rxpair_detections[(det0.rxid, det1.rxid)].append((det0, det1))

    timestamps = {}
    for pair, detections in rxpair_detections.items():
        detections = sorted(detections, key=lambda x: x[0].timestamp) # sort by timestamp
        timestamps[pair] = [d[0].timestamp for d in detections]

    def extract(rxid0, rxid1, timestamp_start, timestamp_stop):
        assert rxid0 < rxid1
        pair = (rxid0, rxid1)
        left = bisect.bisect_left(timestamps[pair], timestamp_start)
        right = bisect.bisect_right(timestamps[pair], timestamp_stop)
        detection_pairs = rxpair_detections[pair][left:right]

        if len(detection_pairs) > 1:
            sdoa = np.array([d[0].soa - d[1].soa for d in detection_pairs])
            is_outlier = stat_tools.is_outlier(sdoa)
            detection_pairs = list(itertools.compress(detection_pairs,
                                                      ~is_outlier)) # select non-outliers

        return detection_pairs

    return extract # return a function?


def estimate_model_quality(model, detection_pairs):
    # TODO: estimate model quality from SNR and/or residuals and/or model
    # covariance matrix and/or model residuals.
    # Alternative names to replace "quality": confidence / beacon SNR
    sqrt_snr0 = np.array([d[0].corr_info.energy / d[0].corr_info.noise
                          for d in detection_pairs])
    sqrt_snr1 = np.array([d[1].corr_info.energy / d[1].corr_info.noise
                          for d in detection_pairs])
    snr = (np.mean(sqrt_snr0**2) + np.mean(sqrt_snr1**2)) / 2

    return snr



def build_model_weighted_poly(detection_pairs, beacon_sdoa,
                              nominal_sample_rate, deg=2):
    if len(detection_pairs) < deg + 1:
        # not enough beacon transmissions
        return None

    soa0 = np.array([d[0].soa for d in detection_pairs])
    soa1 = np.array([d[1].soa for d in detection_pairs])
    soa1at0 = soa1 + np.array(beacon_sdoa)

    def evaluate(det0, det1):
        # # Option 1: weight on min energy
        # weights = [min(d[0].carrier_info.energy,
        #                d[1].carrier_info.energy)
        #            for d in detection_pairs]
        # weights = np.array(weights)
        # weights = weights / np.max(weights)

        # Option 2: weight on mean energy
        # weights = [np.sqrt(d[0].carrier_info.energy**2 +
        #                    d[1].carrier_info.energy**2)
        #            for d in detection_pairs]
        # weights = np.array(weights)
        # weights = weights / np.max(weights)

        # Option 3: weight on "distance" from mobile unit detection
        weights = np.sqrt(1. / (np.abs(soa0 - (det0.soa+det1.soa))))
        weights = weights / np.max(weights)
        weights = np.sqrt(weights)
        weights = (weights + 2) / 3

        # Option 4: hybrid

        coef = np.polyfit(soa1at0, soa0, deg, w=weights)
        fit = np.poly1d(coef)

        return (det0.soa - fit(det1.soa)) / nominal_sample_rate

    return evaluate


def find_nearest_value(list_, value):
    idx = bisect.bisect_left(list_, value)
    if idx > 0 and (idx == len(list_) or
                    abs(value - list_[idx-1]) < abs(value - list_[idx])):
        return idx - 1
    else:
        return idx


def test_find_nearest_value():
    list_ = [5, 10, 15]
    values = [4, 5, 6, 9, 10, 11, 14, 16]
    expected_output = [0, 0, 0, 1, 1, 1, 2, 2]
    nearest = [find_nearest_value(list_, v) for v in values]
    np.testing.assert_equal(nearest, expected_output)


def build_model_nearest(detection_pairs, beacon_sdoa, nominal_sample_rate):
    if len(detection_pairs) < 1:
        # not enough beacon transmissions
        return None

    pairs = sorted(detection_pairs, key=lambda pair: pair[0].timestamp)
    timestamps = [p[0].timestamp for p in pairs]

    def evaluate(det0, det1):
        idx = find_nearest_value(timestamps, det0.timestamp)
        # print(idx, timestamps, det0.timestamp)
        dsoa0 = det0.soa - pairs[idx][0].soa
        dsoa1 = det1.soa - pairs[idx][1].soa
        return (dsoa0 - dsoa1 + beacon_sdoa[idx]) / nominal_sample_rate
        # FIXME: ^ + of - beacon_sdoa?

    return evaluate


def build_model_linear(detection_pairs, beacon_sdoa, nominal_sample_rate):
    if len(detection_pairs) < 2:
        # not enough beacon transmissions
        return None

    pairs = sorted(detection_pairs, key=lambda pair: pair[0].timestamp)

    timestamps = [p[0].timestamp for p in pairs]

    def evaluate(det0, det1):
        high_idx = bisect.bisect_left(timestamps, det0.timestamp)
        if high_idx == len(timestamps):
            high_idx -= 1

        low_idx = high_idx - 1

        # find the nearest beacon transmission from the same beacon
        while (low_idx >= 0 and
                pairs[low_idx][0].txid != pairs[high_idx][0].txid):
            low_idx -= 1

        if low_idx < 0:
            return None

        beacon0 = pairs[low_idx]
        beacon1 = pairs[high_idx]

        weight = ((det0.soa - beacon0[0].soa) /
                  (beacon1[0].soa - beacon0[0].soa))
        tau = ((beacon0[1].soa * (1-weight) + beacon1[1].soa * weight) -
               det1.soa)
        # FIXME: ^ *-1?

        return (tau + beacon_sdoa[high_idx]) / nominal_sample_rate
        # FIXME: ^ + of - beacon_sdoa?

    return evaluate


# default model


def build_model_poly(detection_pairs, beacon_sdoa, nominal_sample_rate, deg=2):
    if len(detection_pairs) < deg + 1:
        # not enough beacon transmissions
        return None

    soa0 = np.array([d[0].soa for d in detection_pairs]) 
    soa1 = np.array([d[1].soa for d in detection_pairs])
    soa1at0 = soa1 + np.array(beacon_sdoa)

    coef = np.polyfit(soa1at0, soa0, deg)
    fit = np.poly1d(coef)
    
    # all_indices = [i for i in range(len(detection_pairs)) if detection_pairs[i][0].txid == 0]
    # soa0_new = [soa0[i] for i in all_indices]
    # soa1_new = [soa1[i] for i in all_indices]
    # soa1at0_new = soa1_new + np.array([beacon_sdoa[i] for i in all_indices])
    # coef = np.polyfit(soa1at0_new, soa0_new, deg)
    # fit = np.poly1d(coef)
    # print([(soa0[f]-fit(soa1[f])-beacon_sdoa[f])/nominal_sample_rate*1e9 for f in range(len(soa0))])   
    # residuals = soa0 - fit(soa1at0)
    # print(np.mean(residuals))

    def evaluate(det0, det1):
        return (det0.soa - fit(det1.soa)) / nominal_sample_rate

    return evaluate

def _dist(vector1, vector2):
    diff = np.array(vector1) - np.array(vector2)
    return np.sqrt(np.sum(diff**2))

default_model = build_model_poly

def estimate_tdoas(detections, matches_dict_values, window_size,
                   beacon_pos, rx_pos, sample_rate,
                   model_builder=default_model, model_params=None):
    if model_params is None:
        model_params = {}

    matches = [list(m) for m in matches_dict_values]

    beacon_matches = []
    for m in matches:
        if len(detections) > 0 and 0 <= m[0] < len(detections):
            index = m[0]
            if detections[index].txid in beacon_pos:
                beacon_matches.append(m)
        else:
            print(f"Invalid index {m[0]} for detections list.")


    mobile_matches = [(i, m) for i, m in enumerate(matches)
                      if detections[m[0]].txid not in beacon_pos]


    def _beacon_tdoa(rxid0, rxid1, beaconid):
        return (_dist(rx_pos[rxid0], beacon_pos[beaconid]) -
                _dist(rx_pos[rxid1], beacon_pos[beaconid])) / SPEED_OF_LIGHT

    tdoa_groups = []
    failures = []

    extractor = make_detection_extractor(detections, beacon_matches)


    for group_idx, group in mobile_matches:
        tdoas = []
        group_timestamp = detections[group[0]].timestamp
        print("Current Group ID: ", str(group_idx), " out of", str(len(mobile_matches)))
        for det0_id, det1_id in itertools.combinations(group, 2):

            if detections[det0_id].rxid > detections[det1_id].rxid:
                det0_id, det1_id = det1_id, det0_id

            if detections[det0_id].rxid in [3, 5] or detections[det1_id].rxid in [3, 5]:
                continue

            det0, det1 = detections[det0_id], detections[det1_id]

            window_start, window_stop = (det0.timestamp - window_size,
                                         det0.timestamp + window_size)
            
            beacon_pairs = extractor(det0.rxid, det1.rxid,
                                     window_start, window_stop)
            

            beacon_tdoa = [_beacon_tdoa(det0.rxid, det1.rxid, b[0].txid)
                           for b in beacon_pairs]
            beacon_sdoa = np.array(beacon_tdoa) * sample_rate

            model = model_builder(beacon_pairs,
                                  beacon_sdoa,
                                  sample_rate,
                                  **model_params)
            if model is None:
                failures.append((det0_id, det1_id))
                continue
            model_quality = estimate_model_quality(model, beacon_pairs)
            tdoa = model(det0, det1)

            # Ignore outliers
            if tdoa is None or abs(tdoa) >= MAX_TDOA:
                failures.append(detections[group[0]].txid)
                continue

            snr0 = (det0.corr_info.energy / det0.corr_info.noise)**2
            snr1 = (det1.corr_info.energy / det1.corr_info.noise)**2
            snr = (snr0 + snr1) / 2

            tdoas.append(TdoaInfo(rx0=det0.rxid,
                                  rx1=det1.rxid,
                                  tdoa=tdoa,
                                  snr=snr,
                                  model_quality=model_quality,
                                  det0_idx=det0_id,
                                  det1_idx=det1_id))

        if len(tdoas) > 0:
            tdoas_array = np.array(tdoas, dtype=TDOA_DTYPE)
            tdoa_groups.append(TdoaGroup(group_id=group_idx,
                                         timestamp=group_timestamp,
                                         tx=det0.txid,
                                         tdoas=tdoas_array))

        ## log file generation

    return tdoa_groups, failures

##########################################################################
from datetime import datetime

start_time_str = '2023-06-15 17:45:00'
end_time_str = '2023-06-15 18:02:00'
# Convert string to datetime object
start_timestamp = int(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S').timestamp())
end_timestamp = int(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S').timestamp())


def estimate_tdoas_w_variances(detections, matches_dict_values, window_size,
                   beacon_pos, rx_pos, sample_rate,
                   model_builder=default_model, model_params=None):
    if model_params is None:
        model_params = {}

    matches = [list(m) for m in matches_dict_values]

    beacon_matches = []
    for m in matches:
        if len(detections) > 0 and 0 <= m[0] < len(detections):
            index = m[0]
            if detections[index].txid in beacon_pos:
                beacon_matches.append(m)
        else:
            print(f"Invalid index {m[0]} for detections list.")


    mobile_matches = [(i, m) for i, m in enumerate(matches)
                      if detections[m[0]].txid not in beacon_pos]


    def _beacon_tdoa(rxid0, rxid1, beaconid):
        return (_dist(rx_pos[rxid0], beacon_pos[beaconid]) -
                _dist(rx_pos[rxid1], beacon_pos[beaconid])) / SPEED_OF_LIGHT

    tdoa_groups = []
    failures = []

    extractor = make_detection_extractor(detections, beacon_matches)


    for group_idx, group in mobile_matches:
        tdoas = []
        group_timestamp = detections[group[0]].timestamp
        if group_timestamp < start_timestamp:
            continue

        # print("Current Group ID: ", str(group_idx), " out of", str(len(mobile_matches)))
        for det0_id, det1_id in itertools.combinations(group, 2):

            if detections[det0_id].rxid > detections[det1_id].rxid:
                det0_id, det1_id = det1_id, det0_id

            det0, det1 = detections[det0_id], detections[det1_id]

            window_start, window_stop = (det0.timestamp - window_size,
                                         det0.timestamp + window_size)
            
            beacon_pairs = extractor(det0.rxid, det1.rxid,
                                     window_start, window_stop)
            

            beacon_tdoa = [_beacon_tdoa(det0.rxid, det1.rxid, b[0].txid)
                           for b in beacon_pairs]
            beacon_sdoa = np.array(beacon_tdoa) * sample_rate

            model = model_builder(beacon_pairs,
                                  beacon_sdoa,
                                  sample_rate,
                                  **model_params)
            if model is None:
                failures.append((det0_id, det1_id))
                continue
            model_quality = estimate_model_quality(model, beacon_pairs)
            tdoa = model(det0, det1)

            # Ignore outliers
            if tdoa is None or abs(tdoa) >= MAX_TDOA:
                failures.append(detections[group[0]].txid)
                continue

            # tdoa_beacons = [model()]

            snr0 = (det0.corr_info.energy / det0.corr_info.noise)**2
            snr1 = (det1.corr_info.energy / det1.corr_info.noise)**2
            snr = (snr0 + snr1) / 2

            tdoas.append(TdoaInfo(rx0=det0.rxid,
                                  rx1=det1.rxid,
                                  tdoa=tdoa,
                                  snr=snr,
                                  model_quality=model_quality,
                                  det0_idx=det0_id,
                                  det1_idx=det1_id))

        if len(tdoas) > 0:
            tdoas_array = np.array(tdoas, dtype=TDOA_DTYPE)
            tdoa_groups.append(TdoaGroup(group_id=group_idx,
                                         timestamp=group_timestamp,
                                         tx=det0.txid,
                                         tdoas=tdoas_array))

        ## log file generation

    return tdoa_groups, failures


def estimate_tdoas_w_logs(detections, matches_dict_values, window_size,
                   beacon_pos, rx_pos, sample_rate,
                   model_builder=default_model, model_params=None):
    if model_params is None:
        model_params = {}

    matches = [list(m) for m in matches_dict_values]

    beacon_matches = [m for m in matches
                      if detections[m[0]].txid in beacon_pos]
    mobile_matches = [(i, m) for i, m in enumerate(matches)
                      if detections[m[0]].txid not in beacon_pos]

    def _beacon_tdoa(rxid0, rxid1, beaconid):
        return (_dist(rx_pos[rxid0], beacon_pos[beaconid]) -
                _dist(rx_pos[rxid1], beacon_pos[beaconid])) / SPEED_OF_LIGHT

    tdoa_groups = []
    failures = []

    extractor = make_detection_extractor(detections, beacon_matches)

    for group_idx, group in mobile_matches:

        if group_idx < 112724 or len(group) < 2: 
            continue

        tdoas = []
        group_timestamp = detections[group[0]].timestamp

        for det0_id, det1_id in itertools.combinations(group, 2):

            if detections[det0_id].rxid > detections[det1_id].rxid:
                det0_id, det1_id = det1_id, det0_id

            det0, det1 = detections[det0_id], detections[det1_id]

            window_start, window_stop = (det0.timestamp - window_size,
                                            det0.timestamp + window_size)
                
            beacon_pairs = extractor(det0.rxid, det1.rxid,
                                        window_start, window_stop)
                
            print("pos of mobile transmission:", str(len(extractor(det0.rxid, det1.rxid, window_start, det0.timestamp))), 
                  " out of ", str(len(beacon_pairs)))
            
            beacon_tdoa = [_beacon_tdoa(det0.rxid, det1.rxid, b[0].txid)
                            for b in beacon_pairs]
            beacon_sdoa = np.array(beacon_tdoa) * sample_rate

            model = model_builder(beacon_pairs,
                                    beacon_sdoa,
                                    sample_rate,
                                    deg = 2)
            if model is None:
                failures.append((det0_id, det1_id))
                continue

            tdoa = model(det0, det1)

            model_poly = build_model_weighted_poly(beacon_pairs,
                                  beacon_sdoa,
                                  sample_rate,
                                  **model_params)
            tdoa_poly = model_poly(det0, det1)

            beacon_tdoas_fitted = [model(f[0], f[1]) for f in beacon_pairs]
            beacon_tdoas_fitted_diff = np.array(beacon_tdoas_fitted)*1e9 - np.array(beacon_tdoa)*1e9
            refit_idx = []
            # for index, value in enumerate(beacon_tdoas_fitted_diff):
            #     if abs(value) < 200:
            #         refit_idx.append(index)
            for index, value in enumerate(beacon_tdoas_fitted_diff):
                if beacon_pairs[index][0].txid == 0:
                    refit_idx.append(index)

            refit_beacon_pairs = [beacon_pairs[f] for f in refit_idx]
            refit_beacon_tdoa = [_beacon_tdoa(det0.rxid, det1.rxid, b[0].txid)
                            for b in refit_beacon_pairs]
            refit_beacon_sdoa = np.array(refit_beacon_tdoa) * sample_rate
            
            if len(refit_beacon_pairs) > 2:
                refit_model = model_builder(refit_beacon_pairs,
                                    refit_beacon_sdoa,
                                    sample_rate,
                                    deg = 2)
                if refit_model is not None:
                    refit_tdoa = refit_model(det0, det1)
                    refit_beacon_tdoas_fitted = [model(f[0], f[1]) for f in refit_beacon_pairs]
                    refit_beacon_tdoas_fitted_diff = np.array(refit_beacon_tdoas_fitted)*1e9 - np.array(refit_beacon_tdoa)*1e9

                else:
                    refit_tdoa = tdoa
            else:
                refit_tdoa = tdoa

            if model is None:
                failures.append((det0_id, det1_id))
                continue

            # Ignore outliers
            if tdoa is None or abs(tdoa) >= MAX_TDOA:
                failures.append((det0_id, det1_id))
                continue

            counter_dict = collections.Counter({i: 0 for i in list(beacon_pos.keys())})
            counter_list = [beacon_pairs[i][0].txid for i in range(len(beacon_pairs))]
            for element in counter_list:
                counter_dict[element] += 1

            tdoas.append((det0.rxid, det1.rxid, tdoa*1e9, refit_tdoa*1e9))
            tdoa_mat = np.zeros((len(group), len(group)))
            unique_rx_ids = np.unique([f[0:2] for f in tdoas])
            for idx, tdoa_data in enumerate(tdoas):
                idx_row = np.where(unique_rx_ids ==tdoa_data[0])[0]
                idx_col = np.where(unique_rx_ids ==tdoa_data[1])[0]
                tdoa_mat[idx_row, idx_col] = tdoa_data[3]
                tdoa_mat[idx_col, idx_row] = -tdoa_data[3]
            Eigenvalues, Eigenvectors = np.linalg.eig(tdoa_mat)

        if len(tdoas) > 2:
            # tdoas_array = np.array(tdoas, dtype=TDOA_DTYPE)
            tdoas_array_log = [(group_idx, group_timestamp, det0.txid) + f for f in tdoas]
            tdoa_groups = tdoa_groups + tdoas_array_log
            
    return tdoa_groups, failures



def estimate_tdoas_timestamps(detections, matches_dict_values, window_size,
                   beacon_pos, rx_pos, sample_rate,
                   model_builder=default_model, model_params=None):
    if model_params is None:
        model_params = {}

    matches = [list(m) for m in matches_dict_values]

    beacon_matches = [m for m in matches
                      if detections[m[0]].txid in beacon_pos]
    mobile_matches = [(i, m) for i, m in enumerate(matches)
                      if detections[m[0]].txid not in beacon_pos]

    def _beacon_tdoa(rxid0, rxid1, beaconid):
        return (_dist(rx_pos[rxid0], beacon_pos[beaconid]) -
                _dist(rx_pos[rxid1], beacon_pos[beaconid])) / SPEED_OF_LIGHT

    tdoa_groups = []
    failures = []

    extractor = make_detection_extractor(detections, beacon_matches)

    for group_idx, group in mobile_matches:
        tdoas = []
        group_timestamp = detections[group[0]].timestamp

        if group_idx < 112536:
            print("skipped", group_idx < 112536)
            continue
        else:
            print("yes")

            for det0_id, det1_id in itertools.combinations(group, 2):
                if detections[det0_id].rxid > detections[det1_id].rxid:
                    det0_id, det1_id = det1_id, det0_id

                det0, det1 = detections[det0_id], detections[det1_id]

                window_start, window_stop = (det0.timestamp - window_size,
                                            det0.timestamp + window_size)
                
                beacon_pairs = extractor(det0.rxid, det1.rxid,
                                        window_start, window_stop)
                

                beacon_tdoa = [_beacon_tdoa(det0.rxid, det1.rxid, b[0].txid)
                            for b in beacon_pairs]
                beacon_sdoa = np.array(beacon_tdoa) * sample_rate

                model = model_builder(beacon_pairs,
                                    beacon_sdoa,
                                    sample_rate,
                                    **model_params)
                if model is None:
                    failures.append((det0_id, det1_id))
                    continue
                model_quality = estimate_model_quality(model, beacon_pairs)

                tdoa = model(det0, det1)

                # Ignore outliers
                if tdoa is None or abs(tdoa) >= MAX_TDOA:
                    failures.append((det0_id, det1_id))
                    continue

                snr0 = (det0.corr_info.energy / det0.corr_info.noise)**2
                snr1 = (det1.corr_info.energy / det1.corr_info.noise)**2
                snr = (snr0 + snr1) / 2

                tdoas.append(TdoaInfo(rx0=det0.rxid,
                                    rx1=det1.rxid,
                                    tdoa=tdoa,
                                    snr=snr,
                                    model_quality=model_quality,
                                    det0_idx=det0_id,                                    det1_idx=det1_id))

            if len(tdoas) > 0:
                tdoas_array = np.array(tdoas, dtype=TDOA_DTYPE)
                tdoa_groups.append(TdoaGroup(group_id=group_idx,
                                            timestamp=group_timestamp,
                                            tx=det0.txid,
                                            tdoas=tdoas_array))
                

    return tdoa_groups, failures



# def load_tdoa_matrix(fname):
#     data = np.loadtxt(fname, dtype=MATRIX_DTYPE)
#     data['tdoa'] /= 1e9
#     return data


def save_tdoa_groups(output, tdoa_groups):
    if isinstance(output, string_types):
        output = open(output, 'w')
    for group in tdoa_groups:
        for tdoa in group.tdoas:
            tdoa = tdoa.copy()
            tdoa['tdoa'] *= 1e9
            print(group.group_id, "%.06f" % group.timestamp, group.tx,
                  *tdoa, file=output)


#############################################################################################

TdoaInfo = collections.namedtuple('TdoaInfo', [
    'rx0', 'rx1', 'tdoa', 'snr', 'model_quality', 'det0_idx', 'det1_idx'])

TdoaGroup = collections.namedtuple('TdoaGroup', [
    'group_id', 'timestamp', 'tx', 'tdoas'])

TDOA_DTYPE = {'names': ('rx0', 'rx1', 'tdoa', 'snr', 'model_quality',
                        'det0_idx', 'det1_idx'),
              'formats': ('i4', 'i4', 'f8', 'f8', 'f8', 'i4', 'i4')}

MATRIX_DTYPE = {'names': ('group_id', 'timestamp', 'tx') + TDOA_DTYPE['names'],
                'formats': ('i4', 'f8', 'i4') + TDOA_DTYPE['formats']}


# def load_tdoa_matrix(fname):
#     data = np.loadtxt(fname, dtype=MATRIX_DTYPE)
#     data['tdoa'] /= 1e9
#     return data

# def groups_to_matrix(groups):
#     data = []
#     for group in groups:
#         group_info = (group.group_id, group.timestamp, group.tx)
#         for tdoa in group.tdoas:
#             row = group_info + tdoa.tolist()
#             data.append(row)
#     return np.array(data, dtype=MATRIX_DTYPE)


# def load_tdoa_groups(fname):
#     matrix = load_tdoa_matrix(fname)
#     tdoa_groups = collections.OrderedDict()
#     for row in matrix:
#         group_id = row['group_id']
#         fields = list(TDOA_DTYPE['names'])
#         if group_id not in tdoa_groups:
#             in_group = matrix['group_id'] == group_id
#             tdoa_groups[group_id] = TdoaGroup(group_id=group_id,
#                                               timestamp=row['timestamp'],
#                                               tx=row['tx'],
#                                               tdoas=matrix[fields][in_group])
#     return tdoa_groups.values()


## replace np.loadtxt for faster loading
def load_tdoa_groups(fname):
    # Initialize an empty list to hold the structured data

    current_group_id = None
    tdoa_groups = []
    tmp_tdoa_group = []

    # Open the file and read lines
    with open(fname, 'r') as fopen:
        lines = fopen.readlines()

    # Process each line
    for line in lines:
        # Split line into components based on whitespace or another delimiter
        components = line.strip().split()  # Adjust split as necessary for your data format
        converted = tuple(
            np.dtype(fmt).type(val) for fmt, val in zip(MATRIX_DTYPE['formats'], components)
        )
        group_id = int(converted[0])

        if group_id != current_group_id:
            # If this is a new group, save the previous group (if it exists) and start a new one
            if current_group_id is not None:  # Avoid appending an empty initial group
                tdoa_groups.append(tmp_tdoa_group)
            tmp_tdoa_group = [converted]
            current_group_id = group_id            
        else:
            tmp_tdoa_group.append(converted)

    # Append the last group after the loop
    if tmp_tdoa_group:
        tdoa_groups.append(tmp_tdoa_group)
    
    return tdoa_groups

#############################################################################################
####################################  SOLVER  ###############################################
#############################################################################################


class EstimationError(Exception):
    pass


def solve_numerically(tdoa_array, rx_pos, dimensions, MAX_DIST):
    """Solve position using the Levenberg-Marquardt minimization algorithm."""
    # TODO: use analytic solution or previous position as initial value

    uniq_rx = np.unique(np.concatenate([tdoa_array['rx0'], tdoa_array['rx1']]))
    

    if len(uniq_rx) <= dimensions :
        raise EstimationError("Underdetermined")

    rx_coords  = np.array([f[0:dimensions] for f in rx_pos.values()])
    min_bounds = np.amin(rx_coords, axis=0) - MAX_DIST[0:dimensions]
    max_bounds = np.amax(rx_coords, axis=0) + MAX_DIST[0:dimensions]

    rx0 = np.array([rx_pos[rxid][0:dimensions] for rxid in tdoa_array['rx0']])
    rx1 = np.array([rx_pos[rxid][0:dimensions] for rxid in tdoa_array['rx1']])

    x0 = (max_bounds + min_bounds) / 2

    def model(pos):
        # position relative to {rx0, rx1}
        pos_rx0, pos_rx1 = rx0 - pos, rx1 - pos
        # distance to {rx0, rx1}
        dist0 = np.linalg.norm(pos_rx0, axis=1)
        dist1 = np.linalg.norm(pos_rx1, axis=1)
        # predicted TDOA (in m)
        predicted_tdoa = dist0 - dist1

        residuals = tdoa_array['tdoa'] * SPEED_OF_LIGHT - predicted_tdoa
        return residuals

    def jac(pos):
        pos_rx0, pos_rx1 = rx0 - pos, rx1 - pos
        dist0 = np.linalg.norm(pos_rx0, axis=1)
        dist1 = np.linalg.norm(pos_rx1, axis=1)
        return pos_rx0 / dist0[:, None] - pos_rx1 / dist1[:, None]

    # print("min_bounds: ", min_bounds)
    # print("max_bounds: ", max_bounds)
    # print("x0: ", x0)
    res = least_squares(model, x0,
                                       jac=jac,
                                       bounds=(min_bounds, max_bounds), verbose=0)

    # TODO: also return residual or a measure of the quality or confidence of
    #       the estimate
    snr_mean = np.mean(tdoa_array['snr'])

    return res, snr_mean, uniq_rx

#############################################################################################

def dop_matrix(pos, rx_pos, rx_pairs):
    rx_pairs = list(rx_pairs)
    pos = np.array(pos)
    rx0 = np.array([np.array(rx_pos[rxid]) for rxid, _ in rx_pairs])
    rx1 = np.array([np.array(rx_pos[rxid]) for _, rxid in rx_pairs])
    pos_rx0, pos_rx1 = rx0 - pos, rx1 - pos
    dist0 = np.linalg.norm(pos_rx0, axis=1)
    dist1 = np.linalg.norm(pos_rx1, axis=1)
    G = pos_rx0 / dist0[:, None] - pos_rx1 / dist1[:, None]
    H_inv = G.T.dot(G)
    try:
        H = np.linalg.inv(H_inv)
    except np.linalg.LinAlgError:
        H = None
    return H


def dop(pos, rx_pos, rx_pairs):
    matrix = dop_matrix(pos, rx_pos, rx_pairs)
    if matrix is None:
        return -1
    return np.sqrt(np.trace(matrix))

#############################################################################################

# # 2D or 3D 

def solve(tdoa_groups, rx_pos, MAX_DIST=[5e3, 5e3, 500]):

    start = time.time()

    results = []
    # tx_rx_pair_results = collections.defaultdict(list)

    # if any([len(rx_pos[f]) < dimensions for f in rx_pos]) or len(rx_pos)<dimensions+1:
    #     print("--Check RX configuration--")
    #     return None       

    # if dimensions == 2:
    #     print("Solve for 2D locations of bats")
    # if dimensions >= 3:
    #     print("Solve for 3D locations of bats")
    
    for group_id, timestamp, tx, tdoas in tdoa_groups:      
        tdoa_dimension = len(tdoas)
        try:
            if tdoa_dimension == 3:
                est_res, snr, uniq_rxs = solve_numerically(tdoas, rx_pos, 2, MAX_DIST)
                rxs_binary_vec = [int(f in uniq_rxs) for f in rx_pos.keys()]
                results.append((group_id, timestamp, tx) + tuple([est_res.x[0], est_res.x[1], None]) + tuple(rxs_binary_vec))
            # dop_est  = dop(est_res.x, rx_pos, rx_pairs)
            elif tdoa_dimension >= 4:
                est_res, snr, uniq_rxs = solve_numerically(tdoas, rx_pos, 3, MAX_DIST)
                rxs_binary_vec = [int(f in uniq_rxs) for f in rx_pos.keys()]
                results.append((group_id, timestamp, tx) + tuple(est_res.x) + tuple(rxs_binary_vec))
            else:
                continue

        except EstimationError as e:
            # print("Failed to estimate group #{}: {}".format(group_id, e))
            continue

    # TODO: apply Kalmin filter or something to average out the position
    #       estimates (move to separate module)
    # for tx_rx_pair in tx_rx_pair_results.keys():
    #     print('num of results of TX-RX Pair ' + str(tx_rx_pair) + ': ' + str(len(tx_rx_pair_results[tx_rx_pair])))

    end = time.time()
    print("Time elapsed solving locations: ", end - start)

    return results

#############################################################################################

def save_est_positions(output, results):
    for position in results:
        fields = list(position)
        fields[1] = "{:.6f}".format(fields[1])  # format timestamp
        print(*fields, file=output)


# def load_positions(fname):
#     num_fields = len(np.genfromtxt(fname, max_rows=1))  # FIXME
#     dtype = {
#         'names': POSITION_INFO_DTYPE['names'][:num_fields],
#         'formats': POSITION_INFO_DTYPE['formats'][:num_fields]
#     }
#     data = np.genfromtxt(fname, dtype=dtype)
#     return data

#############################################################################################