#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "testing allow_resume on multitask_resnet_fpn_masked_embeddings ..."

python main_multitask.py --config config.triplet_segmentation_test_folder.multitask_resnet_fpn_masked_embeddings_test

echo "testing allow_resume on multitask_resnet_fpn_masked_embeddings ..."