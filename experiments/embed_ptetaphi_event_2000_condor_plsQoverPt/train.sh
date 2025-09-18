#!/usr/bin/env bash
set -euo pipefail

# preamble
echo "SHELL=$SHELL"
echo "HOST=$(hostname)"
which python || true
python --version || true
which apptainer || true
apptainer --version || true
which singularity || true
singularity --version || true

# singularity pull --name ./python310.sif docker://python:3.10-slim
# singularity exec ./python310.sif python --version
# singularity exec /ceph/users/atuna/work/gavin_tc_dnn/condor/python310.singularity.sif python --version

echo "python3 --version"
python3 --version

# env?
TOP="/ceph/users/atuna/work/gavin_tc_dnn"
# source ${TOP}/env/bin/activate
# which python
# ls -ltrh /usr/bin/python*
# which python
# env

C_QPT="$1"
C_ETA="$2"
C_PHI="$3"

MODEL="model_weights_ptetaphi_${C_QPT}_${C_ETA}_${C_PHI}.pth"
PDF="plots_ptetaphi_${C_QPT}_${C_ETA}_${C_PHI}.pdf"
MAIN="${TOP}/python/main_physics.py"
DATA="${TOP}/experiments/embed_ptetaphi_event_2000_condor_plsQoverPt"
EPOCHS="2"

echo "${DATA}/env/bin/python --version"
${DATA}/env/bin/python --version

time ${DATA}/env/bin/python ${MAIN} \
     --features_t5 ${DATA}/features_t5.pkl \
     --features_pls ${DATA}/features_pls.pkl \
     --sim_features_t5 ${DATA}/sim_features_t5.pkl \
     --sim_features_pls ${DATA}/sim_features_pls.pkl \
     --model ${MODEL} \
     --c_qpt ${C_QPT} \
     --c_eta ${C_ETA} \
     --c_phi ${C_PHI} \
     --num_epochs ${EPOCHS} \
     --pdf ${PDF}

