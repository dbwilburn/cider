# CIDer: a statistical framework for interpreting CID and HCD fragmentation

Initial release. CIDer is a simple Python3 compatible script that will "CID correct" spectrum libraries acquired by HCD in DLIB format. The conversion is based on a series of linear weights that are packaged as separate pickled parameters. 

Usage: Download the repository, and from the CIDer directory, execute: python CIDer.py /path/to/my_dlib.dlib . This will generate a new DLIB named /path/to/my_dlib_CIDer.dlib .
