EXP_DIR=output/r50_smca_e12

if [ ! -d "output" ]; then
    mkdir output
fi

if [ ! -d "${EXP_DIR}" ]; then
    mkdir ${EXP_DIR}
fi

srun -p dsta \
    --job-name=train \
    --gres=gpu:4 \
    --ntasks=4 \
    --ntasks-per-node=4 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-34 \
    python main.py \
    --batch_size 4 \
    --smca \
    --epochs 12 \
    --lr_drop 10 \
    --output_dir ${EXP_DIR} \
     2>&1 | tee ${EXP_DIR}/detailed_log.txt


