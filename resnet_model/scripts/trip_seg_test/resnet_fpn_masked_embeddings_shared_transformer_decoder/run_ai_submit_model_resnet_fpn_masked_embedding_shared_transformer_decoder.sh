runai submit resnet-fpn-masked-embeddings-shared-transformer-decoder-2\
  -i aicregistry:5000/talabi:mmdet \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --host-ipc \
  --command -- bash /nfs/home/talabi/repositories/triplet_segmentation/resnet_model/scripts/triplet_segmentation_test_folder/resnet_fpn_masked_embeddings_shared_transformer_decoder/resnet_fpn_masked_embedding_shared_transformer_decoder.sh
