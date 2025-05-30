#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet fpn single task verbtarget parrallel decoders dfif lr..."

python main.py --config config.triplet_segmentation_test_folder.singletask_resnet_fpn_verbtarget_parallel_decoders_diff_lr

echo "completed train and test resnet fpn single task verb parallel decoders diff  lr..."