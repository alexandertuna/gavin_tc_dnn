# for DIR in use*; do
for DIR in use_default use_no_phi use_phi_plus_pi use_phi_projection use_pls_deltaphi use_scheduler use_upweight_displaced use_delta_r2_cut_0p2 use_delta_r2_cut_1p0 use_ge50_sim_matching; do
    cd ${DIR}
    time python ../../plot_pca.py --model ../../${DIR}/model_weights.pth
    cd -
done
