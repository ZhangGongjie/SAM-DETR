EXP_DIR=output/r50_e12

if [ ! -d "output" ]; then
    mkdir output
fi

if [ ! -d "${EXP_DIR}" ]; then
    mkdir ${EXP_DIR}
fi

srun -p innova \
    --job-name=train \
    --gres=gpu:4 \
    --ntasks=4 \
    --ntasks-per-node=4 \
    --cpus-per-task=2 \
    --kill-on-bad-exit=1 \
    python main.py \
    --batch_size 4 \
    --epochs 12 \
    --lr_drop 10 \
    --output_dir ${EXP_DIR} \
     2>&1 | tee ${EXP_DIR}/detailed_log.txt




EXP2_DIR=output/r50_e50

if [ ! -d "${EXP2_DIR}" ]; then
    mkdir ${EXP2_DIR}
fi

srun -p innova \
    --job-name=train \
    --gres=gpu:4 \
    --ntasks=4 \
    --ntasks-per-node=4 \
    --cpus-per-task=2 \
    --kill-on-bad-exit=1 \
    python main.py \
    --batch_size 4 \
    --epochs 50 \
    --lr_drop 40 \
    --output_dir ${EXP2_DIR} \
     2>&1 | tee ${EXP2_DIR}/detailed_log.txt
