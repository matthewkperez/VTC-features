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
from utils import xcorr
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



def plot_corr(np_arr,wdir,filename):

    feat_type_dir = os.path.join(wdir, "corr_plot")
    if not os.path.exists(feat_type_dir):
        os.makedirs(feat_type_dir)

    # plot the correlations for sanity
    num_feats = 3
    num_d = np_arr.shape[-1]
    # print(num_d)

    X_final = np.empty([num_feats*num_feats, num_d])
    idx=0
    y_ticks = []
    for i in range(num_feats):
        for j in range(num_feats):
            # plot first 2 dmfcc
            X_final[idx] = np_arr[i,j,:]

            y_ticks.append("({},{})".format(i,j))

            idx+=1

    plt.clf()
    ax = sns.heatmap(X_final)
    ax.set(xlabel='Delay', ylabel='Channels')
    ax.set_yticklabels(y_ticks, rotation=0)
    fig = ax.get_figure()
    fig.savefig(os.path.join(feat_type_dir, "{}.pdf".format(filename)))



def autocorrelation(feats, wdir, filename, auto_corr, first_coef_drop, feat_type):
    # Perform Autocorrelation
    D = int(auto_corr) # 100

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




    ## Sanity check (print corr)
    plot_corr(np_arr=corr_arr_D, wdir=wdir, filename=filename)


    return corr_arr_D

def main():
    # Define what features to extract
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_type', help='feature type: [formant, mfcc, dmfcc]')
    parser.add_argument('audio_file', help='path to audio file (wav)')
    parser.add_argument('feat_dir', help='dir to write features to (output features)')
    parser.add_argument('auto_corr', help='auto_corr nums 81 used in our work')
    parser.add_argument('raw_wdir', help='wdir raw')
    args = parser.parse_args()


    assert args.feat_type in ['formant', 'mfcc', 'dmfcc', 'ddmfcc']

    first_coef_drop = True # if true, drop first mfcc related value (energy constant), if false replace it with RMSE


    # Load features
    if args.feat_type == 'mfcc':
        y, sr = librosa.load(args.audio_file)
        feats = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, hop_length=int(0.010*sr), n_fft=int(0.025*sr))

        # 0th coefficient is constant
        if first_coef_drop:
            feats = feats[1:,:]
        else:
            # repalce with energy
            feats[0] = librosa.feature.rms(y=y, hop_length=int(0.010*sr))

        # cmvn normalize
        feats = feats.astype(float)
        feats = sklearn.preprocessing.scale(feats, axis=1)

    elif args.feat_type == 'dmfcc':
        y, sr = librosa.load(args.audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, hop_length=int(0.010*sr), n_fft=int(0.025*sr))

        # cmvn normalize
        mfcc = mfcc.astype(float)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

        # 0th coefficient is constant
        if first_coef_drop:
            mfcc = mfcc[1:,:]
        else:
            # repalce with energy
            mfcc[0] = librosa.feature.rms(y=y,hop_length=int(0.010*sr))

        feats = librosa.feature.delta(mfcc, order=1)



    elif args.feat_type == 'ddmfcc':
        y, sr = librosa.load(args.audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, hop_length=int(0.010*sr), n_fft=int(0.025*sr))

        # cmvn normalize
        mfcc = mfcc.astype(float)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

        # 0th coefficient is constant
        if first_coef_drop:
            mfcc = mfcc[1:,:]
        else:
            # replace with energy
            mfcc[0] = librosa.feature.rms(y=y, hop_length=int(0.010*sr))
        feats = librosa.feature.delta(mfcc, order=2)

    elif args.feat_type == 'formant':
        # PFP
        formants = pfp.formants_at_interval(args.audio_file)
        formants = np.transpose(formants) # get it into formant (num_channels, num_frames)
        feats = formants[1:,:]
        
        # normalize cmvn
        feats = feats.astype(float)
        feats = sklearn.preprocessing.scale(feats, axis=1)



    if first_coef_drop and cmvn_bool:
        feat_type_dir = os.path.join(args.feat_dir,"cmvn_drop_first_coef", args.feat_type)
        raw_feat_type_dir = os.path.join(args.raw_wdir, "cmvn_drop_first_coef", args.feat_type)
    elif first_coef_drop and not cmvn_bool:
        feat_type_dir = os.path.join(args.feat_dir,"drop_first_coef", args.feat_type)
        raw_feat_type_dir = os.path.join(args.raw_wdir, "drop_first_coef", args.feat_type)
    elif not first_coef_drop and cmvn_bool:
        feat_type_dir = os.path.join(args.feat_dir,"cmvn_rmse_first_coef", args.feat_type)
        raw_feat_type_dir = os.path.join(args.raw_wdir, "cmvn_rmse_first_coef", args.feat_type)
    else:
        feat_type_dir = os.path.join(args.feat_dir,"rmse_first_coef", args.feat_type)
        raw_feat_type_dir = os.path.join(args.raw_wdir, "rmse_first_coef", args.feat_type)



    if not os.path.exists(feat_type_dir):
        os.makedirs(feat_type_dir)

    if not os.path.exists(raw_feat_type_dir):
        os.makedirs(raw_feat_type_dir)

    # get uttname
    filename=args.audio_file.split("/")[-1].split(".")[0]

    # Compute correlations 
    corr_data = autocorrelation(feats=feats, wdir=feat_type_dir, filename=filename, auto_corr=args.auto_corr, first_coef_drop=first_coef_drop, feat_type=args.feat_type)

    # add length in seconds to filename (for SS files where we need to filter out some data by length)
    seconds = int(sox.file_info.duration(args.audio_file))

    # Save correlation features to 'feats dir'
    save_file = os.path.join(feat_type_dir, "{}_{}.npy".format(filename,seconds))
    np.save(save_file, corr_data)

    # Save raw features to 'raw_feat_type_dir'
    save_raw_file = os.path.join(raw_feat_type_dir, "{}_{}.npy".format(filename,seconds))
    np.save(save_raw_file, feats)

    # # print mfcc
    plt.clf()
    sns.palplot(sns.diverging_palette(240, 10, n=9))
    ax = sns.heatmap(feats[:,:300], cmap=sns.color_palette("coolwarm", as_cmap=True))
    ax.invert_yaxis()
    fig = ax.get_figure()
    fig.savefig(os.path.join(feat_type_dir, "{}_3s.png".format(filename)))



if __name__ == '__main__':
    main()