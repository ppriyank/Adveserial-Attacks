#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=5:00:00
#SBATCH --mem=50000
#SBATCH --job-name=pp1953
#SBATCH --mail-user=pp1953p@nyu.edu
#SBATCH --output=slurm_%j.out

. ~/.bashrc
module load gcc/9.1.0
module load anaconda3/5.3.1
conda activate pathak
conda install -n pathak nb_conda_kernels



module load gcc/9.1.0
conda activate pathak
cd 
cd /scratch/pp1953/cml/Adveserial-Attacks/paper2018/
python main2.py -t "test is a test" --out-name=0  --ctc-weight=1 --db-weight=1 --l2-weight=0.0005



