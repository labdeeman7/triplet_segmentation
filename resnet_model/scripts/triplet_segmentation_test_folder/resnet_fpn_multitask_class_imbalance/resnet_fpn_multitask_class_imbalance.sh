#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

python main.py --config config.triplet_segmentation_test_folder.multitask_resnet_fpn_class_imbalance

echo "completed train and test resnet fpn multitask class imbalance..."