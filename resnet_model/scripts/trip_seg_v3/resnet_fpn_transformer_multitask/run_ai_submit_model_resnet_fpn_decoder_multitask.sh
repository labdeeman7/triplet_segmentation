runai submit resnet-fpn-decoder-multitask\
  -i aicregistry:5000/talabi:mmdet \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --host-ipc \
  --command -- bash /nfs/home/talabi/repositories/triplet_segmentation/resnet_model/scripts/trip_seg_v3/resnet_fpn_transformer_multitask/resnet_fpn_decoder_multitask.sh