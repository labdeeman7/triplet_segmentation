#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet fpn single task verbtarget_v2_5 ..."

python main.py --config config.triplet_segmentation_v2_5_dataset.singletask_resnet_fpn_verbtarget_v2_5

echo "completed train and test resnet fpn singlet task verbtarget_v2_5..."