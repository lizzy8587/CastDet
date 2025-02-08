DEVICES_ID=2
exp="oriented-rcnn_r50-fpn_20k_visdronezsd_base-set"

CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/train.py \
    projects/CastDetv2/configs/$exp.py

python projects/CastDetv2/tools/merge_weights.py \
    --clip_path checkpoints/RemoteCLIP-RN50.pt \
    --base_path work_dirs/$exp/iter_20000.pth \
    --save_path work_dirs/$exp/merged_castdet_init_iter20k.pth \
    --base_model faster-rcnn


exp="visdrone_step2_castdet_12b_10k_oriented"

CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/train.py \
    projects/CastDetv2/configs/$exp.py

CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/test.py \
    projects/CastDetv2/configs/$exp.py \
    work_dirs/$exp/iter_10000.pth \
    --work-dir work_dirs/$exp/dior_test
