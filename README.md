Description:
This repository contains code for extracting Vocal Tract Coordination (VTC) features. These features are computed by applying a cross correlation function to different channels in the acoustic signal (mfcc, dmfcc, formants, etc.). The resulting feature set we call Full Vocal Tract Coordination (FVTC) features, which have dimensionality CxCxT, where C is the number of channels and T is the number of time delays to consider. Previous works, have experimented with dimensionality reduction techniques such as eigendecomposition, there is a boolean flag which can be toggled to produce these Eigen Vocal Tract Coordination (EVTC) features. EVTC features have dimensionality of CxT.

Directions:
Script takes in the arguments listed below. It will create 2 directories, which will contain the VTC features as well as the raw acoustic features. 

Argument Parser Descriptions:
-feat_type = Either Mfcc, dmfcc, ddmfcc, or formant
-audio_file = Wav file to have features extracted from
-delays = Number of time delays to consider from cross correlation function
-vtc_dir = Directory to write VTC (FVTC/EVTC) features to
-raw_dir = Directory to write raw acoustic features to

