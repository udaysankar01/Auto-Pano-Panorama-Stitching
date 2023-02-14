# Phase 1 - AutoPano using classical techniques

First download all the required data from the source in Phase1/Data/ directory. 

Use this comand to open the directory for phase 1:

`cd ./Phase1`

To run the panorama stitching for Train Set 1:

`python Wrapper.py --ImagesFolder Data/Train/Set1`

To run the panorama stitching for Train Set 2:

`python Wrapper.py --ImagesFolder Data/Train/Set2`

To run the panorama stitching for Train Set 3:

`python Wrapper.py --ImagesFolder Data/Train/Set3`

To run the panorama stitching for Test Set 1:

`python Wrapper.py --ImagesFolder Data/Test/TestSet1`

To run the panorama stitching for Test Set 2:

`python Wrapper.py --ImagesFolder Data/Test/TestSet2`

To run the panorama stitching for Test Set 3:

`python Wrapper.py --ImagesFolder Data/Test/TestSet3`

To run the panorama stitching for Test Set 4:

`python Wrapper.py --ImagesFolder Data/Test/TestSet4`

All the results will be stored in the folder shown below:

`./Results/`





# Phase 2 - Deep Learning Approach

First download all the required data from the source in Phase2/Data/ directory. Also, add a Checkpoints folder in Phase2/ directory.

Generate data using data_generation file, while having the Train and Val images in Data, inside Phase2/ directory.

To train the supervised model, run `python3 Train_supervised.py`

To train the unsupervised model, run `python3 Train_unsupervised.py`

To test the supervised model, uncomment lines 16 and 28 in Test.py and run `python3 Test.py`

To test the unsupervised model, uncomment lines 17 and 29 in Test.py and run `python3 Test.py`
