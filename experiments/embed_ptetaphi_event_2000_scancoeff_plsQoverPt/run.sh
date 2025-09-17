EPOCHS=200
C_QPT=1.0
C_ETA=1.0
C_PHI=1.0
MODEL="model_weights_ptetaphi_${C_QPT}_${C_ETA}_${C_PHI}.pth"
time python ../../python/main_physics.py --model ${MODEL} --c_qpt ${C_QPT} --c_eta ${C_ETA} --c_phi ${C_PHI} --num_epochs ${EPOCHS} | tee log_physics_${C_QPT}_${C_ETA}_${C_PHI}.txt
