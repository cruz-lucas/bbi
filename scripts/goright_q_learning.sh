#!/bin/bash
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00                       # Run for DD-HH:MM
#SBATCH --job-name=goright_q_learning
#SBATCH --output=%x-%j.out

export $(xargs <.env)
cd $SLURM_TMPDIR
git clone git@github.com:cruz-lucas/bbi.git
cd ./bbi

module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install -r requirements.txt --no-index

make q-learning

wandb sync tmp/wandb/ --sync-all --clean-force

deactivate