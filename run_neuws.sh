#!/bin/bash
#SBATCH --job-name=neuws_recon
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-metzler
#SBATCH --qos=vulcan-high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=22G
#SBATCH --time=06:00:00

# Print job information
echo "Starting job $SLURM_JOB_ID on node $(hostname)"
echo "Running in directory: $(pwd)"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# Activate virtual environment
source /fs/nexus-scratch/bhetwal/gsplat_env/bin/activate  # <-- adjust path to your venv if different

# Load any necessary modules (optional, depending on cluster setup)
# module load cuda/12.1
# module load python/3.10

# Run Python script
python recon_exp_data.py \
    --static_phase \
    --use_gsplat \
    --num_t 100 \
    --root_dir /fs/nexus-scratch/bhetwal/NeuWS_data/static_objects_static_aberrations/dog_esophagus_0.5diffuser/Zernike_SLM_data  \
    --scene_name dog_esophagus_gaussian \
    --phs_layers 4 \
    --num_epochs 4000 \
    --num_gaussians 30000 \
    --vis_freq 500 \
    --gs_model_type 2dgs \
   # --gauss_lr 1e-4 \
# Deactivate environment
deactivate

echo "Job finished at $(date)"
