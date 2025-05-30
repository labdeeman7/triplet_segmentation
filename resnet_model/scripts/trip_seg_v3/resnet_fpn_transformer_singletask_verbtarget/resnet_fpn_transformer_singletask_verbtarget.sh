#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet fpn single task verbtarget with decoder ..."

python main.py --config config.triplet_segmentation_v3_dataset.singletask_resnet_fpn_transformer_decoder

echo "completed train and test resnet fpn singlet task verbtarget with decoder..."