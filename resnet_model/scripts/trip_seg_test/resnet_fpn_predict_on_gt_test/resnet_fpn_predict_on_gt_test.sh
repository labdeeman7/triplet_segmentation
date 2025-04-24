#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet fpn predict on gt test ..."

python main.py --config config.triplet_segmentation_test_folder.multitask_resnet_fpn_predict_on_gt_test

echo "completed train and test resnet fpn predict on gt test ..."