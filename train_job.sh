#!/bin/bash
#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1:accelerator_model=a100
#PBS -l walltime=05:30:00
#PBS -A DialSys

cd $PBS_O_WORKDIR

module load hpc_basis uv/0.10.2 gcc/13.2.0 Openssl/1.1.1t Tcl/8.6.11 Tk/8.6.13 Python/3.12.3 CUDA/12.6.1 tmux/3.3a

echo "Loaded Modules:"
module list
echo "Allocated GPU:"
nvidia-smi

uv run train_gpt.py