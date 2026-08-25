#!/bin/bash
# command line parameter: train(default)/test/clean
# train: downloads all data and
# test: only the test set is downloaded

MODE=${1:-train}

# at the moment the dataset is still hosted on kaggle
# need kaggle.json config file in /home/user/.kaggle at first use, which contains the configuration of your kaggle account
# fresh download plus unzip (change unzip path to the correct download place)
if [[ $MODE = "train" ]]; then
    echo "Selected train and test data"
    kaggle datasets download -d 'sorour/38cloud-cloud-segmentation-in-satellite-images'
    echo "Extracting Cloud38 part to obpmark-ml/semantic_segmentation/data"
    unzip -qq -n 38cloud-cloud-segmentation-in-satellite-images.zip -d ../../data
    rm 38cloud-cloud-segmentation-in-satellite-images.zip

    kaggle datasets download -d 'sorour/95cloud-cloud-segmentation-on-satellite-images'
    echo "Extracting Cloud95 part to obpmark-ml/semantic_segmentation/data"
    unzip -qq -n 95cloud-cloud-segmentation-on-satellite-images.zip -d ../../data
    rm 95cloud-cloud-segmentation-on-satellite-images.zip

    echo "Cleaning up files not needed.."
    pushd ../../data
    mv 38-Cloud_training/train_blue/* 95-cloud_training_only_additional_to38-cloud/train_blue_additional_to38cloud/
    mv 38-Cloud_training/train_red/* 95-cloud_training_only_additional_to38-cloud/train_red_additional_to38cloud/
    mv 38-Cloud_training/train_green/* 95-cloud_training_only_additional_to38-cloud/train_green_additional_to38cloud/
    mv 38-Cloud_training/train_gt/* 95-cloud_training_only_additional_to38-cloud/train_gt_additional_to38cloud/
    mv 38-Cloud_training/train_nir/* 95-cloud_training_only_additional_to38-cloud/train_nir_additional_to38cloud/
    rm -r 38-Cloud_training
    rm -r 38-Cloud_Training_Metadata_Files
    rm bibtex.txt
    rm training_patches_38-cloud_nonempty.csv
    mv 95-cloud_training_only_additional_to38-cloud 95-Cloud_training
    popd

    echo "Done!"
elif [[ $MODE = "test" ]]; then
    echo "Selected only test data"
    kaggle datasets download -d 'sorour/38cloud-cloud-segmentation-in-satellite-images'

    echo "Extracting to obpmark-ml/semantic_segmentation/data"
    unzip -qq -n 38cloud-cloud-segmentation-in-satellite-images.zip -d ../../data
    rm 38cloud-cloud-segmentation-in-satellite-images.zip

    echo "Cleaning up files not needed.."
    pushd ../../data
    rm -r 38-Cloud_training
    rm -r 38-Cloud_Training_Metadata_Files
    rm bibtex.txt
    rm training_patches_38-cloud_nonempty.csv
    popd

    echo "Done!"
elif [[ $MODE = "clean" ]]; then
    echo "Cleaning up obpmark-ml/semantic_segmentation/data folder"
    rm -r ../../data/*
    echo "Done!"
else
    echo "Specify which data you want to download: train/test/clean"
fi
