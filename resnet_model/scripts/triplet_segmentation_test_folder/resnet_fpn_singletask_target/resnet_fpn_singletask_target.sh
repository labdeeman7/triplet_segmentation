#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet fpn single task target ..."

python main_singletask.py --config config.triplet_segmentation_test_folder.singletask_resnet_fpn_target

echo "completed train and test resnet fpn singlet task target..."