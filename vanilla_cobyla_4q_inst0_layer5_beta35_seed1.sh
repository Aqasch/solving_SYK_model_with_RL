#!/bin/bash
#
#SBATCH --job-name="NNforQAS"
#SBATCH --partition=gpu
#SBATCH --clusters ukko
#SBATCH -o vanilla_cobyla_SYK_4q_inst0_layer5_cnn_finite_beta35.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --time=48:00:00
#SBATCH -G 3
#SBATCH --priority=5100000
#SBATCH --begin=2024-06-13T17:18:00
#
#
srun python main_syk_finite.py --seed 1 --config vanilla_cobyla_SYK_4q_inst0_layer5_cnn_finite_beta35 --experiment_name "finalize/"