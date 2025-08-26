FI75="/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p75_pLSdeltaPhiChargeXYZ_event_2000.root"
FI50="/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p50_pLSdeltaPhiChargeXYZ_event_2000.root"

cd use_default
time python ../../../python/main.py --num_epochs 2 --input ${FI75} --test_size 0.95
cd -

# for DIR in use_no_phi use_phi_plus_pi use_phi_projection use_pls_deltaphi use_scheduler; do
#     cd ${DIR}
#     time python ../../../python/main.py --num_epochs 2 --input ${FI75} --test_size 0.95 --${DIR}
#     cd -
# done

# cd use_upweight_displaced
# time python ../../../python/main.py --num_epochs 2 --input ${FI75} --test_size 0.95 --upweight_displaced 1.0
# cd -

# cd use_delta_r2_cut_0p2
# time python ../../../python/main.py --num_epochs 2 --input ${FI75} --test_size 0.95 --delta_r2_cut 0.2
# cd -

# cd use_delta_r2_cut_1p0
# time python ../../../python/main.py --num_epochs 2 --input ${FI75} --test_size 0.95 --delta_r2_cut 1.0
# cd -

# cd use_ge50_sim_matching
# time python ../../../python/main.py --num_epochs 2 --input ${FI50} --test_size 0.95
# cd -
