EXP_DIR=output/r50_ms_smca_e50

if [ ! -d "output" ]; then
    mkdir output
fi

if [ ! -d "${EXP_DIR}" ]; then
    mkdir ${EXP_DIR}
fi

srun -p cluster_name \
    --job-name=SAM-DETR \
    --gres=gpu:8 \
    --ntasks=8 \
    --ntasks-per-node=8 \
    --cpus-per-task=2 \
    --kill-on-bad-exit=1 \
    python main.py \
    --batch_size 2 \
    --smca \
    --multiscale \
    --epochs 50 \
    --lr_drop 40 \
    --output_dir ${EXP_DIR} \
     2>&1 | tee ${EXP_DIR}/detailed_log.txt
