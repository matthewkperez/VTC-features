Description:
This repository contains code for extracting Vocal Tract Coordination (VTC) features. These features are computed by applying a cross correlation function to different channels in the acoustic signal (mfcc, dmfcc, formants, etc.). The resulting feature set we call Full Vocal Tract Coordination (FVTC) features, which have dimensionality CxCxT, where C is the number of channels and T is the number of time delays to consider. Previous works, have experimented with dimensionality reduction techniques such as eigendecomposition, there is a boolean flag which can be toggled to produce these Eigen Vocal Tract Coordination (EVTC) features. EVTC features have dimensionality of CxT.


