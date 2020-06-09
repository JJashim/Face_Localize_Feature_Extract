
### Face Localization using MTCNN model and Face Feature Extraction using Facenet

Description:

Requirement.txt - contains list of packages needed to be installed

facenet/src/facenet.py: Required to load (facenet) model to perform facial feature extraction.  
facenet/facenet_bk.py: the original file -- (incase) replace it if the existing facenet.py creates any issues. Also, It can be used for understanding image batch processing.   
facenet/align: code and model to perform MTCNN face localization.

LICENECE.md - MIT Licence --Required to be present.

Models/* - Contains model path to perform facial feature extraction.  
face_localize_feature_extract.py - execute to perform face localization and feature extraction.


Howto:

Required python 3.7 (more specific v3.7.3) -- pls check with other python 3 versions too.  
Required packages can be installed using 'pip install -r requirements.txt'  
$: python face_localize_feature_extract.py. 

Add-on:

Print Input folder and Model paths at the start.
Print Output Folder and CSV path at End.

This work is influenced by [Facenet](https://github.com/davidsandberg/facenet)

Face Feature Extraction Model can be downloaded from [here](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-). The downloaded model to be unzipped under model directory.

