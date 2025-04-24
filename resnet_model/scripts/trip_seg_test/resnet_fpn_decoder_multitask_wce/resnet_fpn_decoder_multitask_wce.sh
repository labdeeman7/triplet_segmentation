#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet fpn multitask decoder wce..."

python main.py --config config.triplet_segmentation_test_folder.multitask_resnet_fpn_transformer_decoder_wce

echo "completed train and test resnet fpn multitask decoder wce..."