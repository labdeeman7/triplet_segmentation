runai submit threetasks-parallel-decoders-low-lr-big-image\
  -i aicregistry:5000/talabi:mmdet \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --host-ipc \
  --command -- bash /nfs/home/talabi/repositories/triplet_segmentation/resnet_model/scripts/triplet_segmentation_test_folder/resnet_fpn_threetask_parallel_decoders_low_lr_big_image/resnet_fpn_threetasks_parallel_decoders_low_lr_big_image.sh
