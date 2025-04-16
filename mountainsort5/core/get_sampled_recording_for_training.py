import spikeinterface as si
import numpy as np
from typing import Literal
import math
import warnings

import tqdm

def get_sampled_recording_for_training(
    recording: si.BaseRecording, *,
    training_duration_sec: float,
    mode: Literal['initial', 'uniform'] = 'initial'
) -> si.BaseRecording:
    """Get a sampled recording for the purpose of training

    Args:
        recording (si.BaseRecording): SpikeInterface recording object
        training_duration_sec (float): Duration of the training in seconds
        mode (str): 'initial' or 'uniform'

    Returns:
        si.BaseRecording: SpikeInterface recording object
    """
    if training_duration_sec * recording.sampling_frequency >= recording.get_num_frames():
        # if the training duration is longer than the recording, then just use the entire recording
        return recording
    if mode == 'initial':
        traces = recording.get_traces(start_frame=0, end_frame=int(training_duration_sec * recording.sampling_frequency))
    elif mode == 'uniform':
        # use chunks of 10 seconds
        chunk_size = int(recording.sampling_frequency * min(10, training_duration_sec))
        # the number of chunks depends on the training duration
        num_chunks = int(np.ceil(training_duration_sec * recording.sampling_frequency / chunk_size))
        chunk_sizes = [chunk_size for i in range(num_chunks)]
        chunk_sizes[-1] = int(training_duration_sec * recording.sampling_frequency - (num_chunks - 1) * chunk_size)
        if num_chunks == 1:
            # if only 1 chunk, then just use the initial chunk
            traces = recording.get_traces(start_frame=0, end_frame=int(training_duration_sec * recording.sampling_frequency))
        else:
            # the spacing between the chunks
            spacing = int((recording.get_num_frames() - np.sum(chunk_sizes)) / (num_chunks - 1))
            traces_list: list[np.ndarray] = []
            tt = 0
            for i in range(num_chunks):
                start_frame = tt
                end_frame = int(start_frame + chunk_sizes[i])
                traces_list.append(recording.get_traces(start_frame=start_frame, end_frame=end_frame))
                tt += int(chunk_sizes[i] + spacing)
            traces = np.concatenate(traces_list, axis=0)
    else:
        raise Exception('Invalid mode: ' + mode) # pragma: no cover

    rec = si.NumpyRecording(
        traces_list=[traces],
        sampling_frequency=recording.sampling_frequency,
        channel_ids=recording.get_channel_ids()
    )
    rec.set_channel_locations(recording.get_channel_locations())
    return rec


def _get_continuous_clean_window_ids(clean_mask_, *, chunksize_limit):
    """
    Get continuous clean window indices.
    
    Args:
        clean_mask_ (np.ndarray) 1d boolean array. 1 means clean and 0 means noisy.
        chunksize_limit : (int | None) if not None, then set an upper limit on the size of each detected continuous chunk
    
    Returns:
        list of tuples: each 2-tuple (start_window_index, duration_in_windows) describes a duration of continuous clean data
    """
    ret = []
    i = 0
    nwins = len(clean_mask_)
    if chunksize_limit:
        check_size = lambda l: (l<=chunksize_limit)
    else:
        check_size = lambda _: True
    while i < nwins:
        j = i
        while ((j<nwins) and (clean_mask_[j]) and check_size(j-i)):
            j += 1
        if j > i:
            ret.append((i, j-i)) # a clean duration
        i = j + 1 # start from next window
    return ret

def _get_data_sampled_windows(rec_: si.BaseRecording, sampled_windows_, window_len_in_frames_):
    # get traces
    traces_list = []
    selected_frame_begends = []
    for start_window_id, len_in_windows in sampled_windows_:
        beg_frame = start_window_id * window_len_in_frames_
        end_frame = beg_frame + len_in_windows*window_len_in_frames_
        traces_list.append(rec_.get_traces(start_frame=beg_frame, end_frame=end_frame))
        selected_frame_begends.append([beg_frame, end_frame])
    traces = np.concatenate(traces_list, axis=0)
    rec = si.NumpyRecording(
        traces_list=[traces],
        sampling_frequency=rec_.sampling_frequency,
        channel_ids=rec_.get_channel_ids()
    )
    rec.set_channel_locations(rec_.get_channel_locations())
    return rec, np.array(selected_frame_begends)

def get_selected_sampled_recording_for_training(
    recording: si.BaseRecording, *,
    training_ratio: float,
    max_training_duration_sec: float,
    window_len_sec : float,
    artifact_mask : np.ndarray
) -> si.BaseRecording:
    """Get a sampled recording for the purpose of training

    Args:
        recording (si.BaseRecording): SpikeInterface recording object
        training_ratio (float): Ratio of the training (0.0 - 1.0)
        max_training_duration_sec (float): maximum training duration in seconds.
            Need to limit this because all training set is loaded in mem at the same time
        window_len_samples (int): Duration of each window in seconds
        artifact_mask (np.ndarray): boolean array of shape (n_windows,) that indicates whther each window is noise contaminated.

    Returns:
        si.BaseRecording: SpikeInterface recording object
        
        np.ndarray: (n_chunks, 2) beg and end frames of sampled data chunks
    """
    fs = recording.sampling_frequency
    assert artifact_mask.dtype == bool, "`artifact_mask` must be boolean ndarray"

    window_len_samples = int(fs * window_len_sec)
    windows_avail_mask = ~artifact_mask
    max_training_nwwindows = int(round(max_training_duration_sec/window_len_sec))
    training_nwindows = min(int(np.sum(windows_avail_mask)*training_ratio), max_training_nwwindows)

    if np.sum(windows_avail_mask) <= training_nwindows:
        warnings.warn("All clean data are used as training set.")
        sampled_windows = _get_continuous_clean_window_ids(windows_avail_mask, chunksize_limit=None)
        rec, selected_frame_begends = _get_data_sampled_windows(recording, sampled_windows, window_len_samples)
        return rec, selected_frame_begends

    # determine which windows to take
    sampled_windows = []
    sampled_windows_cnt = 0
    # #### prioritize getting continuous clean chunks
    # #### if there are whole chunks of unpolluted data; use them first.
    # #### Otherwise fall back to smaller chunks, eventually take single windows.
    # #### there is a tradeoff between continuity of samples and uniformed-ness of the sampling
    # taken_nwindows = 0
    # chunk_size_samples = int(fs * chunk_size_seconds)
    clean_chunk_sizes_sec = [10, 10, 5, 5, 2, 2]
    for chunk_size_sec in clean_chunk_sizes_sec:
        print("Finding continuous clean chunks of %d seconds..."%(chunk_size_sec))
        windows_wanted_thisround = training_nwindows-sampled_windows_cnt
        chunk_size_sec = min(chunk_size_sec, windows_wanted_thisround*window_len_sec)
        chunk_size_win = int(math.ceil(chunk_size_sec/window_len_sec))
        nchunks_wanted = (windows_wanted_thisround)//chunk_size_win
        # find continuous clean windowes of input signal 
        clean_windows = _get_continuous_clean_window_ids(windows_avail_mask, chunksize_limit=chunk_size_win)
        clean_chunks = list(filter(lambda x: x[1]>=chunk_size_win, clean_windows))
        # TODO whether to use ceil to avoid sampling too much
        sample_chunk_every = int(math.ceil(len(clean_chunks) / nchunks_wanted))
        if sample_chunk_every >= 1:
            for (beg_window_id, len_in_windows) in tqdm.tqdm(clean_chunks[::sample_chunk_every]):
                sampled_windows.append((beg_window_id, len_in_windows))
                sampled_windows_cnt += len_in_windows
                windows_avail_mask[beg_window_id:(beg_window_id+len_in_windows)] = 0
                if sampled_windows_cnt >= training_nwindows:
                    break
        # if enough windows are sampled, then skip the next alternative chunk size
        if sampled_windows_cnt >= training_nwindows:
            print("Found enough clean chunks for training")
            break
    # if still yet to complete the sampling, take single windows
    while (sampled_windows_cnt < training_nwindows):
        print("Already sampled %d/%d windows"%(sampled_windows_cnt, training_nwindows))
        windows_wanted_thisround = training_nwindows-sampled_windows_cnt
        windows_avail_ids = np.where(windows_avail_mask)[0]
        n_win_avail = len(windows_avail_ids)
        sample_window_every = int(math.ceil(n_win_avail / windows_wanted_thisround))
        for window_id in windows_avail_ids[::sample_window_every]:
            sampled_windows.append((window_id, 1))
            sampled_windows_cnt += 1
            windows_avail_mask[window_id] = 0
            if sampled_windows_cnt >= training_nwindows:
                break
        if sampled_windows_cnt >= training_nwindows:
            break
        elif np.sum(windows_avail_mask)==0:
            warnings.warn("All clean data are used as training set.")
            warnings.warn("Not able to sampled desired number of data")
            break
    print("Loading selected chunks into memory")
    rec, selected_frame_begends = _get_data_sampled_windows(recording, sampled_windows, window_len_samples)
    print("Loading done")
    return rec, selected_frame_begends
