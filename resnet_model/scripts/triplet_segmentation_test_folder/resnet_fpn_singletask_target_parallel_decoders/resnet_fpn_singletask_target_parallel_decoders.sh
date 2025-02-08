#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet fpn single task target parrallel decoders..."

python main.py --config config.triplet_segmentation_test_folder.singletask_resnet_fpn_target_parallel_decoders

echo "completed train and test resnet fpn single task target parallel decoders..."