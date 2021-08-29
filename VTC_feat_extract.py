'''
By Matthew Perez
Extract VTC correlation features in python
For more details about features:
https://ieeexplore.ieee.org/document/9054323
'''

import argparse
import librosa
import numpy as np
import pandas as pd
from scipy import signal
import scipy
import math
import os
import praat_formants_python as pfp
import sklearn


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import sox

def xcorr(x, y, normed=True, detrend=False, maxlags=10):
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    
    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))
    
    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c

def autocorrelation(feats, filename, delays):
    # Perform Autocorrelation
    D = int(delays)

    num_channels = feats.shape[0]
    corr_arr = np.empty([num_channels,num_channels,D])
    corr_arr_D = np.empty([num_channels,num_channels,D+1])


    # N_sources_i
    for i in range(num_channels):
        feat_i = feats[i,:]

        # N_sources_j
        for j in range(num_channels):
            feat_j = feats[j,:]


            # Compute delays altogether
            lags, c = xcorr(x=feat_i, y=feat_j, maxlags=D)

            # isolate correlation, add to feature
            corr_coefs_py = c[D:] # c = [-D,D]. [D:] => [0,D]
            corr_arr_D[i][j] = corr_coefs_py



    return corr_arr_D

def main():
    # Define what features to extract
    parser = argparse.ArgumentParser()
    parser.add_argument('-feat_type', help='feature type: [formant, mfcc, dmfcc, dmfcc]')
    parser.add_argument('-audio_file', help='path to audio file (wav)')
    parser.add_argument('-delays', help='time delays for correlation 81 used in our work')
    parser.add_argument('-vtc_dir', help='dir to write features to (output features)')
    parser.add_argument('-raw_dir', help='dir to write raw wavforms')
    args = parser.parse_args()


    assert args.feat_type in ['formant', 'mfcc', 'dmfcc', 'ddmfcc']

    first_coef_drop = True # if true, drop first mfcc related value (energy constant), if false replace it with RMSE


    # Load features
    if args.feat_type == 'mfcc':
        y, sr = librosa.load(args.audio_file)
        feats = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, hop_length=int(0.010*sr), n_fft=int(0.025*sr))
        # cmvn normalize
        feats = feats.astype(float)
        feats = sklearn.preprocessing.scale(feats, axis=1)
        # 0th coefficient is constant
        feats = feats[1:,:]

    elif args.feat_type == 'dmfcc':
        y, sr = librosa.load(args.audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, hop_length=int(0.010*sr), n_fft=int(0.025*sr))
        
        # cmvn normalize
        mfcc = mfcc.astype(float)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
        mfcc = mfcc[1:,:]

        # 0th coefficient is constant
        feats = librosa.feature.delta(mfcc, order=1)

    elif args.feat_type == 'ddmfcc':
        y, sr = librosa.load(args.audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, hop_length=int(0.010*sr), n_fft=int(0.025*sr))

        # cmvn normalize
        mfcc = mfcc.astype(float)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
        # 0th coefficient is constant
        mfcc = mfcc[1:,:]

        feats = librosa.feature.delta(mfcc, order=2)

    elif args.feat_type == 'formant':
        # PFP
        formants = pfp.formants_at_interval(args.audio_file)
        formants = np.transpose(formants) # get it into formant (num_channels, num_frames)
        feats = formants[1:,:]
        
        # normalize cmvn
        feats = feats.astype(float)
        feats = sklearn.preprocessing.scale(feats, axis=1)


    # Create directories for 
    FVTC_dir = os.path.join(args.vtc_dir,"FVTC", args.feat_type)
    EVTC_dir = os.path.join(args.vtc_dir, "EVTC", args.feat_type)
    raw_feat_dir = os.path.join(args.raw_dir, args.feat_type)

    # Make directories
    if not os.path.exists(FVTC_dir):
        os.makedirs(FVTC_dir)
    if not os.path.exists(EVTC_dir):
        os.makedirs(EVTC_dir)
    if not os.path.exists(raw_feat_dir):
        os.makedirs(raw_feat_dir)

    # get uttname
    filename=args.audio_file.split("/")[-1].split(".")[0]

    # Compute correlations 
    FVTC_data = autocorrelation(feats=feats, filename=filename, delays=args.delays)
    # Save FVTC
    FVTC_save_file = os.path.join(FVTC_dir, "{}.npy".format(filename))
    np.save(FVTC_save_file, FVTC_data)


    # Compute EVTC and save EVTC
    FVTC_data = np.swapaxes(FVTC_data, 0,2)
    EVTC_data, _ = np.linalg.eigh(FVTC_data)
    EVTC_data = np.swapaxes(EVTC_data, 0,1)
    EVTC_save_file = os.path.join(EVTC_dir, "{}.npy".format(filename))
    np.save(EVTC_save_file, EVTC_data)


    # Save raw features to 'raw_feat_type_dir'
    save_raw_file = os.path.join(raw_feat_dir, "{}.npy".format(filename))
    np.save(save_raw_file, feats)



if __name__ == '__main__':
    main()