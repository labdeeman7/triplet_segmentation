runai submit resnet-fpn-singletask-verbtarget-v2-5\
  -i aicregistry:5000/talabi:mmdet \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --host-ipc \
  --command -- bash /nfs/home/talabi/repositories/triplet_segmentation/resnet_model/scripts/triplet_seg_v2_5/resnet_fpn_singletask_verbtarget_v2_5/resnet_fpn_singletask_verbtarget_v2_5.sh
