runai submit resnet-fpn-singletask-verbtarget-3\
  -i aicregistry:5000/talabi:mmdet \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --host-ipc \
  --command -- bash /nfs/home/talabi/repositories/triplet_segmentation/resnet_model/scripts/trip_seg_test/resnet_fpn_singletask_verbtarget/resnet_fpn_singletask_verbtarget.sh
