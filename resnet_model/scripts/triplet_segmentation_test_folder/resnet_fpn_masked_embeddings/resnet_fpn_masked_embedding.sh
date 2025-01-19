#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test multitask_resnet_fpn_masked_embeddings ..."

python main.py --config config.triplet_segmentation_test_folder.multitask_resnet_fpn_masked_embeddings

echo "completed train and test multitask_resnet_fpn_masked_embeddings ..."