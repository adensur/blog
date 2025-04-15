#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2           # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=8   # Use gpus-per-node instead of gres=gpu
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# activate conda env
echo "Slurm job id "$SLURM_JOB_ID
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/traindata/maksim/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/traindata/maksim/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/traindata/maksim/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/traindata/maksim/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate lightly 

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
python tumor_pretrain.py --data datasets/pidray_converted/train/all_train_images2 --epochs 1000 --output out/ssl11 --batch-size 512 --num-nodes 2 --num-gpus 8
