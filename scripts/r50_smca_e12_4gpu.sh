EXP_DIR=output/r50_smca_e12

if [ ! -d "output" ]; then
    mkdir output
fi

if [ ! -d "${EXP_DIR}" ]; then
    mkdir ${EXP_DIR}
fi

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --batch_size 4 \
    --smca \
    --epochs 12 \
    --lr_drop 10 \
    --output_dir ${EXP_DIR} \
     2>&1 | tee ${EXP_DIR}/detailed_log.txt

     
