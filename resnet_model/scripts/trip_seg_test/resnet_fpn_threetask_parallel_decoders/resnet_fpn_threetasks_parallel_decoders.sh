#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet fpn threetasks parrallel decoders..."

python main.py --config config.triplet_segmentation_test_folder.threetask_resnet_fpn_parallel_decoders

echo "completed train and test resnet fpn threetasks parallel decoders..."