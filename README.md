# Patient Monitor Reader

Software for training an optical character recognition model for digits, and running a recognition system to extract numeric
values directly from video of a medical patient monitor. Watch "Patient_Monitor_Demo.mp4" for instructions on how to use.

## Data sources

Char74 dataset: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
MNIST: http://yann.lecun.com/exdb/mnist/
Google Streetview Digits dataset: http://ufldl.stanford.edu/housenumbers/

## Overview

Machine learning model was trained using Char74. So far this has been sufficient for good performance on test videos. The
original model was trained on MNIST augmented with various scaling factors and blurs. The machine learning model is trained
by running "models/run_model.py", and it loads pickled data files that are generated using "process_raw_data.py". Note that raw image datasets
are not included in this repository, and must be downloaded separately from the links above. "process_raw_data.py" was merely copied
from a directory on my local machine containing raw image data, and serves as a template for reference. It will have to be modified
to suit your specific file structure once raw image data has been downloaded. "logs.txt" document the parameters used to train
individual models on each dataset. These models are included in the folders corresponding to their respective datasets.

In order to run the patient monitor, simply run "monitor.py". In order to change the video source, change the argument in lines
18 and 169 to the appropriate filepath leading to the new video file, or the input port for your specific camera (most likely 0 or 1).
Numerical data extracted from each video file is saved in "results" as a csv file - the file name can be changed by modifying line
32 in "monitor.py". Line 35 controls region-of-interest configurations for specific patient monitor setups. If the first element of the array
is set to True, the configuration drawn out on screen will be saved in "configurations". If set to False, a saved configuration will be used
from configurations. The name of the configuration to be saved or retrieved is set in line 33. 
