EXP_DIR=output/r50_dc5_e12

if [ ! -d "output" ]; then
    mkdir output
fi

if [ ! -d "${EXP_DIR}" ]; then
    mkdir ${EXP_DIR}
fi

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --batch_size 1 \
    --dilation \
    --epochs 12 \
    --lr_drop 10 \
    --output_dir ${EXP_DIR} \
     2>&1 | tee ${EXP_DIR}/detailed_log.txt

