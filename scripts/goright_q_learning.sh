#!/bin/bash
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00                       # Run for DD-HH:MM
#SBATCH --job-name=goright_q_learning
#SBATCH --output=%x-%j.out

module load python/3.10 gcc arrow/17.0.0
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install -r requirements.txt
make q-learning

deactivate