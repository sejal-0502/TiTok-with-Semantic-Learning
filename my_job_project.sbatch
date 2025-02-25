#!/bin/bash
#SBATCH --job-name=my_job_project                         # Job name
#SBATCH --output=output_project.txt                       # Standard output
#SBATCH --error=error_project.txt                         # Standard error
#SBATCH --time=24:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=lmbhiwi_gpu-rtx2080        # Partition name
#SBATCH --ntasks=1                                # Number of tasks
#SBATCH --cpus-per-task=4                         # Number of CPUs per task
#SBATCH --mem=32G                                  # Memory per node
#SBATCH --gres=gpu:2
#SBATCH --mail-type=END,FAIL                      # Email notifications
#SBATCH --mail-user=sejalmutakekar@gmail.com # Email address

# Execute your program
srun python main.py --base configs/vqgan_baseline.yaml -t True --n_gpus 2 --n_nodes 1 --name project_100 --checkpoint_dir /work/dlclarge2/mutakeks-storage_titok/project_100

# srun python token_compression/tools/compute_codebook_usage.py --config_path /work/dlclarge2/mutakeks-storage_titok/vqgan123_50_2/checkpoints/vqgan_logs/config.yaml --ckpt_path /work/dlclarge2/mutakeks-storage_titok/vqgan123_50_2/last.ckpt --compute_rFID_score