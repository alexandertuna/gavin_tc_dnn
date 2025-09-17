EPOCHS=50
# C_QPT=1.0
# C_ETA=1.0
# C_PHI=1.0
for C_QPT in 0.25 0.5 1.0 2.0 4.0; do
    for C_ETA in 0.25 0.5 1.0 2.0 4.0; do
        for C_PHI in 0.25 0.5 1.0 2.0 4.0; do

            MODEL="model_weights_ptetaphi_${C_QPT}_${C_ETA}_${C_PHI}.pth"

            time python ../../python/main_physics.py --model ${MODEL} --c_qpt ${C_QPT} --c_eta ${C_ETA} --c_phi ${C_PHI} --num_epochs ${EPOCHS} | tee log_physics_${C_QPT}_${C_ETA}_${C_PHI}.txt

        done
    done
done

