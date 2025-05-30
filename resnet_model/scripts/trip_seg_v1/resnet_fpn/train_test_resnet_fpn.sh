#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet fpn ..."

python main.py --config config.triplet_segmentation_test_folder.multitask_resnet_fpn

echo "completed train and test resnet fpn ..."