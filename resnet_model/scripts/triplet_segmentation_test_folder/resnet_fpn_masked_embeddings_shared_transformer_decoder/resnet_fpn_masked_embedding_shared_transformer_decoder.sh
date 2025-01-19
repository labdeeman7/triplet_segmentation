#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/triplet_segmentation/resnet_model

echo "train and test resnet_fpn_masked_embedding_shared_transformer_decoder ..."

python main.py --config config.triplet_segmentation_test_folder.multitask_resnet_fpn_masked_embeddings_shared_transformer_decoder

echo "completed train and test resnet_fpn_masked_embedding_shared_transformer_decoder ..."