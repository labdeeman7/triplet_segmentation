#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "test predict only ..."

python main.py --config config.triplet_segmentation_test_folder.singletask_resnet_fpn_verbtarget_class_imbalance_addressed

echo "completed predict only ..."