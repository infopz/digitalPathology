#!/bin/bash

#SBATCH --job-name=fix_feature_extraction_her2
#SBATCH --output=/homes/gcasari/reggio_projects/digital_pathology/sbatch/logs/fix_feature_extraction_her2.log
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=bolelli_synthetic
#SBATCH --constraint="gpu_RTX5000_16G|gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"


echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_NAME} started on ${HOSTNAME}"

start_time=$(date +%s)

echo "Using Python from: $(which python)"
echo "Script started at: $(date '+%Y-%m-%d %H:%M:%S')"

# Execute inference
echo "Train set feature extraction"

PYTHONPATH=. python her2_test/extract_uni_features.py \
    --manifest /work/bolelli_synthetic/reggio_data/datasets/ihc4bc/splits_her2/train/manifest.csv \
    --output-root /work/bolelli_synthetic/reggio_data/datasets/ihc4bc/feat_extracted/uni2_her2 \
    --weights /work/bolelli_synthetic/reggio_data/model_weights/uni2-h/pytorch_model.bin

PYTHONPATH=. python her2_test/extract_uni_features.py \
    --manifest /work/bolelli_synthetic/reggio_data/datasets/ihc4bc/splits_her2/val/manifest.csv \
    --output-root /work/bolelli_synthetic/reggio_data/datasets/ihc4bc/feat_extracted/uni2_her2 \
    --weights /work/bolelli_synthetic/reggio_data/model_weights/uni2-h/pytorch_model.bin

PYTHONPATH=. python her2_test/extract_uni_features.py \
    --manifest /work/bolelli_synthetic/reggio_data/datasets/ihc4bc/splits_her2/test/manifest.csv \
    --output-root /work/bolelli_synthetic/reggio_data/datasets/ihc4bc/feat_extracted/uni2_her2 \
    --weights /work/bolelli_synthetic/reggio_data/model_weights/uni2-h/pytorch_model.bin

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_NAME} finished."
echo "Runtime: ${runtime} seconds ($((runtime / 60)) min)"