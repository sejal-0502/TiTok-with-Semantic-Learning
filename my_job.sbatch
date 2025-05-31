#!/bin/bash
#SBATCH --job-name=my_job                         # Job name
#SBATCH --output=output.txt                       # Standard output
#SBATCH --error=error.txt                         # Standard error
#SBATCH --time=24:00:00                           # Time limit (hh:mm:ss)cv_env/my_job.sbatch
#SBATCH --partition=lmbhiwidlc_gpu-rtx2080  # Partition name
#SBATCH --ntasks=1                                # Number of tasks
#SBATCH --cpus-per-task=4                         # Number of CPUs per task
#SBATCH --mem=32G                                  # Memory per node
#SBATCH --gres=gpu:2
#SBATCH --mail-type=END,FAIL                      # Email notifications


# Execute your program
srun python -m torch.distributed.launch --nproc_per_node=2 \
    titok_project/token_compression/main.py \
    --base titok_project/token_compression/configs/vqgan_baseline.yaml \
    -t True --n_gpus 2 --n_nodes 1 --name titok_2 \
    --checkpoint_dir /work/dlclarge2/mutakeks-storage_titok/titok_2

# Resume training 
# srun python -m torch.distributed.launch --nproc_per_node=2 \
#     titok_project/token_compression/main.py \
#     --base titok_project/token_compression/configs/vqgan_baseline.yaml \
#     -t True --n_gpus 2 --resumes /work/dlclarge2/mutakeks-storage_titok/arch_100_5/checkpoints/checkpoints/checkpoints/checkpoints/last.ckpt

# Calculating FID score
# srun export PYTHONPATH=$PYTHONPATH:/home/mutakeks/titok_project/token_compression
# srun python titok_project/token_compression/tools/compute_codebook_usage.py --config_path /work/dlclarge2/mutakeks-storage_titok/arch_100/config.yaml --ckpt_path /work/dlclarge2/mutakeks-storage_titok/arch_100/checkpoints/checkpoints/last.ckpt --compute_rFID_score