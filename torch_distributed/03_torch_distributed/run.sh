set -e # set fail on fail

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

python main.py