# CIDer: a statistical framework for interpreting CID and HCD fragmentation

![alt text](https://pubs.acs.org/na101/home/literatum/publisher/achs/journals/content/jprobs/2021/jprobs.2021.20.issue-4/acs.jproteome.0c00964/20210326/images/medium/pr0c00964_0007.gif)

CIDer is a simple Python3 compatible script that will "CID correct" spectrum libraries acquired by HCD in DLIB format. The conversion is based on a series of linear weights that were learned across multiple NCE settings (6 values spanning 20 to 35) with interpolations for all intermediate values. Details of the training and inferred features can be found in the associated manuscript (https://pubs.acs.org/doi/abs/10.1021/acs.jproteome.0c00964).

Usage: Download the repository, and from the CIDer directory, execute: python CIDer.py (--nce nce_setting) /path/to/my_dlib.dlib (--output output.dlib). 

By default if no additional commands are provided, CIDer will generate a new DLIB named /path/to/my_dlib_CIDer.dlib based on HCD NCE 30. However, HCD NCE tuning appears to be highly instrument specific, and we suggest that users empirically calibrate their particular instrument's NCE settings relative to those employeed by ProteomeTools (whose data was used for model training). This can be performed using the CE calibration tool in Prosit (https://www.proteomicsdb.org/prosit/) or through comparison of spectral similarity between observed and predicted CID spectra (as performed in the manuscript).
