#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet_fpn_trainable_embedding ..."

python main.py --config config.triplet_segmentation_test_folder.multitask_resnet_fpn_learnable_embeddings

echo "completed train and test resnet_fpn_trainable_embedding ..."