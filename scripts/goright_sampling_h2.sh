#!/bin/bash
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=50
#SBATCH --time=1-00:00                       # Run for DD-HH:MM
#SBATCH --job-name=goright_sampling_h2
#SBATCH --output=%x-%j.out

module load python/3.10 gcc arrow/17.0.0
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install -e .
make sampling-2

deactivate