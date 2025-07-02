# ln -sf /ceph/users/atuna/work/gavin_tc_dnn/python/pls_t5_embed.root ./LSTNtuple.ttbarPU200.noembed.root

for TAG in noembed default match75; do  

    # # TAG="noembed"
    # # TAG="default"
    # # TAG="match75"

    SAMPLE="ttbarPU200"
    NTUPLE="LSTNtuple.${SAMPLE}.${TAG}.root"
    NUMDEN="LSTNumDen.${SAMPLE}.${TAG}.root"
    PLOT="TC_duplrate_etacoarsezoom.pdf"

    # time lst_make_tracklooper -mC
    # lst -i PU200RelVal -n 500 -l -s 32 -v 1 -o LSTNtuple.${SAMPLE}.${TAG}.root
    # time createPerfNumDenHists -i ${NTUPLE} -o ${NUMDEN}
    time python3 ../efficiency/python/lst_plot_performance.py -t ${TAG} --sample_name ${SAMPLE} ${NUMDEN}
    cp -a performance/${TAG}_*-ttbarPU200/mtv/var/${PLOT} ${TAG}_${PLOT}

done

