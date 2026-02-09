#!/bin/bash -l
#SBATCH --job-name=loso_revalexo_x3d_m
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=hpc_account_name
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=1
#SBATCH --output=loso_x3dm_%j.out
#SBATCH --error=loso_x3dm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user.name@email.com
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=6000M

# Initialize conda
source /data/leuven/370/vsc37079/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate multimodal-env

# Change to working directory
cd /path/to/codebase

# Send a notification email that job has started
mail -s "[LOSO X3D-M]: STARTED JOB ${SLURM_JOB_ID}" user.name@email.com <<< "LOSO X3D-M model job started on ${SLURM_NODELIST}"

# Print job information
echo "================================================"
echo "LOSO X3D-M Models"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Number of CPUs: ${SLURM_NTASKS}"
echo "GPU count: $(nvidia-smi --list-gpus | wc -l)"
echo "================================================"

# X3D-M model
echo ""
python loso_multi_horizon.py --base-config configs/loso/revalexo/x3d_m_loso.yaml

# Print completion information
echo "================================================"
echo "Job finished at: $(date)"
echo "================================================"
