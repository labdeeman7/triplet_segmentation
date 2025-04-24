runai submit resnet-fpn-decoder-multitask-wce\
  -i aicregistry:5000/talabi:mmdet \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --host-ipc \
  --command -- bash /nfs/home/talabi/repositories/triplet_segmentation/resnet_model/scripts/triplet_segmentation_test_folder/resnet_fpn_decoder_multitask_wce/resnet_fpn_decoder_multitask_wce.sh
