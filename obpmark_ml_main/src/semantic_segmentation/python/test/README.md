# Using the test data of Cloud95

In the current developement stage, the benchmark itself is not fully defined yet. For the testing of the models accuracy
on unseen testing data, some of the scripts provided by the [authors of the Cloud95 dataset](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset/tree/master/evaluation) where adapted and can be used.

The original matlab script was adapted to octave to stick with freely available open source tools. The original matlab 
script is also provided. Use __create_cloudmasks.py__ (uses tensorflow at the moment) to infer the test data and save 
the results. Can be quite memory extensive.