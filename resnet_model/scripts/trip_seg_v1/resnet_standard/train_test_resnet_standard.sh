#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet standard ..."

python main.py --config config.triplet_segmentation_test_folder.multitask_resnet

echo "completed train and test resnet standard ..."