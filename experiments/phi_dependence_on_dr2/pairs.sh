FI75="/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p75_pLSdeltaPhiChargeXYZ_event_2000.root"

# for DR2 in 0.02 0.05 0.1 0.2 0.4 0.6 0.8 1.0; do
for DR2 in 0.3 0.5 0.7 0.9 0.02 0.05 0.1 0.2 0.4 0.6 0.8 1.0; do

    DIR="event_2000_dr2_norm_${DR2}"
    mkdir -p ${DIR}
    cd ${DIR}
    time python ../../../python/main.py --input ${FI75} --test_size 0.95 --delta_r2_cut ${DR2} --num_epochs 1
    rm -f features_* model_*
    cd -

    # DIR="event_2000_dr2_proj_${DR2}"
    # mkdir -p ${DIR}
    # cd ${DIR}
    # time python ../../../python/main.py --input ${FI75} --test_size 0.95 --delta_r2_cut ${DR2} --num_epochs 1 --use_phi_projection
    # rm -rf features_* model_*
    # cd -

done

