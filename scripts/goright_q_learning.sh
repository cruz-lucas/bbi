#!/bin/bash
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00                       # Run for DD-HH:MM
#SBATCH --job-name=goright_q_learning
#SBATCH --output=%x-%j.out

cd $SLURM_TMPDIR
git clone git@github.com:cruz-lucas/bbi.git
cd ./bbi

module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install -r requirements.txt --no-index

export $(xargs <$PROJECT/bbi/.env)
make q-learning

deactivate