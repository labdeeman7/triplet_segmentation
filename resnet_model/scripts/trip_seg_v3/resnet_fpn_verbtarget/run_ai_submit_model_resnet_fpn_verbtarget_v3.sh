runai submit resnet-fpn-singletask-verbtarget-v3\
  -i aicregistry:5000/talabi:mmdet \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --host-ipc \
  --command -- bash /nfs/home/talabi/repositories/triplet_segmentation/resnet_model/scripts/trip_seg_v3/resnet_fpn_verbtarget/resnet_fpn_verbtarget_v3.sh
