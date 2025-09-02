# gavin_tc_dnn

This is a repo for playing with the T5/pLS embedding training and networks. It is entirely based on Gavin's notebook:

https://github.com/SegmentLinking/cmssw/blob/master/RecoTracker/LSTCore/standalone/analysis/DNN/embed_train.ipynb

Please refer to that notebook for the official implementation.

## Datasets

The "official" ttbar PU200 LSTNtuple for training and plotting is `pls_t5_embed.root`, which is copied from Cornell onto the `uaf` machines here:

```bash
-rw-r--r-- 1 atuna atuna 3.1G Jun  2 16:04 /ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed.root
```

A similar dataset can be remade with these commands:

```bash
# Tested with CMSSW_15_1_0_pre4
lst_make_tracklooper -mcCd;
lst -i PU200RelVal -n 500 -l -s 32 -v 1 -o pls_t5_embed.root;
```

I found it useful to make LSTNtuples with additional branches in the TTree (pLS dphi and xyz); using different track truth-matching criteria (50% instead of 75%); and using a different input file (event_2000). They are located here:

```bash
-rw-r--r-- 1 atuna atuna 3.2G Aug  7 17:38 /ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p75_pLSdeltaPhiChargeXYZ.root
-rw-r--r-- 1 atuna atuna 3.2G Aug 13 00:03 /ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p50_pLSdeltaPhiChargeXYZ.root
-rw-r--r-- 1 atuna atuna 6.4G Aug 17 11:24 /ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p75_pLSdeltaPhiChargeXYZ_event_2000.root
-rw-r--r-- 1 atuna atuna 6.4G Aug 18 09:47 /ceph/users/atuna/work/gavin_tc_dnn/data/pls_t5_embed_0p50_pLSdeltaPhiChargeXYZ_event_2000.root
```

The following trackingNtuples are also useful for LSTNtuple under different circumstances:

```bash
# For recreating pls_t5_embed.root exactly:
/ceph/cms/store/user/evourlio/LST/samples/CMSSW_12_2_0_pre2/RelValTTbar_14TeV_CMSSW_12_5_0_pre3/event_1000.root

# For testing with more statistcs:
/ceph/cms/store/user/evourlio/LST/samples/CMSSW_12_2_0_pre2/RelValTTbar_14TeV_CMSSW_12_5_0_pre3/event_2000.root

# For running the cmssw pipeline to produce track efficiency, duplicate rate, and fake rate:
/ceph/cms/store/user/evourlio/LST/samples/CMSSW_12_2_0_pre2/RelValTTbar_14TeV_CMSSW_12_5_0_pre3/event_3000.root
```
