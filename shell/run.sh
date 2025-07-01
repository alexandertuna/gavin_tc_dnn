NTUPLE="pls_t5_embed.root"
# ln -sf /ceph/users/atuna/work/gavin_tc_dnn/python/pls_t5_embed.root ./pls_t5_embed.root

TAG="default"
# TAG="match75"
SAMPLE="ttbarPU200"
NUMDEN="LSTNumDen.${SAMPLE}.${TAG}.root"

time createPerfNumDenHists -i ${NTUPLE} -o ${NUMDEN}
time python3 ../efficiency/python/lst_plot_performance.py -t ${TAG} --sample_name ${SAMPLE} ${NUMDEN}
