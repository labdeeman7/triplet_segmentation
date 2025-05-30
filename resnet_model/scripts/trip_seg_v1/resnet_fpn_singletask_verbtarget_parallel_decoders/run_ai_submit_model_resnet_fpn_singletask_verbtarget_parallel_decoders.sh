runai submit verbtarget-parallel-decoders-2\
  -i aicregistry:5000/talabi:mmdet \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --host-ipc \
  --command -- bash /nfs/home/talabi/repositories/triplet_segmentation/resnet_model/scripts/triplet_segmentation_test_folder/resnet_fpn_singletask_verbtarget_parallel_decoders/resnet_fpn_singletask_verbtarget_parallel_decoders.sh
