#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test four task moe..."

python main.py --config config.triplet_segmentation_test_folder.fourtask_moe

echo "completed train and test four task moe..."