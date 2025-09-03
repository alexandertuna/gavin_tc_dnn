FI75="/ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p75_pLSdeltaPhiChargeXYZ_event_2000.root"
TR2000="/ceph/cms/store/user/evourlio/LST/samples/CMSSW_12_2_0_pre2/RelValTTbar_14TeV_CMSSW_12_5_0_pre3/event_2000.root"

time python ../../python/main.py --num_epochs 2 --input ${FI75}
rm -f pairs*pkl model_weights.pth
time python ../../python/get_sim_features.py --tracking_ntuple ${TR2000}
