#!/bin/bash
#SBATCH -J hello_world
# Please check paths (directories have to exist beforehand):
#SBATCH -e /work/scratch/kurse/kurs00079/ng33rete/new_approach/visualize_pred_dist.err
#SBATCH -o /work/scratch/kurse/kurs00079/ng33rete/new_approach/visualize_pred_dist.out
#
# CPU specification
#SBATCH -n 1 # 1 process
#SBATCH -c 8 # 4 CPU cores per process
# can be referenced as $SLURM_CPUS_PER_TASK?~@~K in the "payload" part
#SBATCH --mem-per-cpu=8192 # Hauptspeicher in MByte pro Rechenkern
#SBATCH -t 24:00:00 # in hours:minutes, or '#SBATCH -t 10' - just minutes
#SBATCH -A kurs00079
#SBATCH -p kurs00079
#SBATCH --reservation=kurs00079
# GPU specification
#SBATCH --gres=gpu:v100:1 # 1 GPUs of type NVidia "Volta 100"
# can be referenced down below as $SLURM_GPUS_ON_NODE
# -------------------------------
# your job's "payload" in form of commands to execute, eg.
# specification from OMP_NUM_THREADS depends on your program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# for checking whether and which GPUs have been allocated
# (output appears in the "#SBATCH -e" file specified above):
nvidia-smi 1>&2
# if your program supports this way of getting told how many GPUs to use:
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE
./run.sh
EXITCODE=$?
# any cleanup and copy commands:
...
# end this job script with precisely the exit status of your scientific program above:
exit $EXITCODE