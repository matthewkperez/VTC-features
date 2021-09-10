Description:  
This repository contains code for extracting Vocal Tract Coordination (VTC) features. These features are computed by applying a cross correlation function to different channels in the acoustic signal (mfcc, dmfcc, formants, etc.). The resulting feature set we call Full Vocal Tract Coordination (FVTC) features, which have dimensionality CxCxT, where C is the number of channels and T is the number of time delays to consider. Previous works, have experimented with dimensionality reduction techniques such as eigendecomposition, there is a boolean flag which can be toggled to produce these Eigen Vocal Tract Coordination (EVTC) features. EVTC features have dimensionality of CxT.

More details about our specific implementation can be found in our paper here: https://www.isca-speech.org/archive/pdfs/interspeech_2021/perez21_interspeech.pdf

Directions:  
Script takes in the arguments listed below. It will create 2 directories, which will contain the VTC features as well as the raw acoustic features. 

Argument Parser Descriptions:  
-feat_type = Either Mfcc, dmfcc, ddmfcc, or formant  
-audio_file = Wav file to have features extracted from  
-delays = Number of time delays to consider from cross correlation function  
-vtc_dir = Directory to write VTC (FVTC/EVTC) features to  
-raw_dir = Directory to write raw acoustic features to  

If you found this helpful please consider citing and checking out the following literature:  

[1] Perez, Matthew, et al. "Articulatory Coordination for Speech Motor Tracking in Huntington Disease}}." Proc. Interspeech 2021 (2021): 1409-1413.

[2] Huang, Zhaocheng, Julien Epps, and Dale Joachim. "Exploiting vocal tract coordination using dilated CNNs for depression detection in naturalistic environments." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.

[3] Williamson, James R., et al. "Vocal biomarkers of depression based on motor incoordination." Proceedings of the 3rd ACM international workshop on Audio/visual emotion challenge. 2013.

[4] Williamson, James R., et al. "Detecting depression using vocal, facial and semantic communication cues." Proceedings of the 6th International Workshop on Audio/Visual Emotion Challenge. 2016.

[5] Talkar, Tanya, et al. "Acoustic Indicators of Speech Motor Coordination in Adults With and Without Traumatic Brain Injury}}." Proc. Interspeech 2021 (2021): 21-25.

[6] Williamson, James R., et al. "Tracking depression severity from audio and video based on speech articulatory coordination." Computer Speech & Language 55 (2019): 40-56.
