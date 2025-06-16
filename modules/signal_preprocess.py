import numpy as np

from scipy import signal
from scipy.ndimage import gaussian_filter

# =========== Helper Layers ========================================================================
# ========== BASIC PREPROCESSING ===========================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # sos = signal.butter(order, [lowcut, highcut], 'bandpass', fs=fs, output='sos')
    # y = signal.sosfilt(sos, data)

    b, a =  signal.butter(order, highcut, fs=fs, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def impute(ts, sr=1000, tr=500):
    idx = [i*tr for i in range(len(ts)) if not np.isnan(ts[i])]
    vals = [ts[i] for i in range(len(ts)) if not np.isnan(ts[i])]
    interp_vals = np.interp(np.arange(len(ts)*tr), idx, vals)
    
    r = sr
    interp_vals = np.array([interp_vals[i] for i in range(len(ts)*tr) if i % r == 0])
                  
    return interp_vals

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def basic_preproc(
    ts, 
    sr=1000, # source sampling rate
    tr=125, # target sampling rate
    l_pass=0.1, # low frequency pass
    h_pass=32, # high frequency pass
    outlier_p=0.95, # threshold for outlier
    smooth_c=0.02,
):  
    # remove outlier
    diffs = np.abs(ts[1:] - ts[:-1])
    max_diff = np.quantile(diffs, outlier_p)
    ts_clear = np.array([_ for _ in ts])
    ts_outliers = diffs > max_diff
    ts_outliers = np.append(ts_outliers, False)
    ts_clear[ts_outliers] = np.nan

    # if h_pass < sr // 2: # optional
    #     # impute missing value
    #     ts_clear = impute(ts_clear, sr=sr, tr=sr)
    
    #     # bandpass filter, remove frequency out of bound
    #     ts_clear = butter_bandpass_filter(ts_clear, l_pass, h_pass, sr, order=4)
    
    # down sample to target sampling rate
    ts_clear = impute(ts_clear, sr=sr, tr=tr)

    # detrend, remove linear shift
    ts_clear = signal.detrend(ts_clear)

    # smooth
    ts_clear = gaussian_filter(ts_clear, sigma=tr*smooth_c)

    # normalize (optional)
    ts_clear /= np.mean(np.abs(ts_clear))
    
    return ts_clear

def preproc_all(all_tss, ss, ts=65, lc=0.1, hc=128, outlier_p=0.95): 
    """
    input:
    all_tss: np.array, shape (C, L), C is the number of channels, L is the length of each channel
    ss: int, source sampling rate, e.g., 65

    output:
    new_tss: np.array, shape (C, L'), 
    C is the number of channels, L' is the length of each channel after preprocessing (resample)
    """
 
    C, L = all_tss.shape
    new_tss = None

    for j in range(C):
        clean_tss = basic_preproc(
                all_tss[j], 
                sr=ss, 
                tr=ts, 
                l_pass=lc,
                h_pass=hc,
                outlier_p=outlier_p
            )
        if new_tss is None:
            new_tss = np.zeros((C, len(clean_tss)))

        new_tss[j] = clean_tss

    return new_tss