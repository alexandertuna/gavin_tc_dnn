# for DIR in dr2_*; do
#     cd ${DIR}
#     cp -a model_weights.pth ../event_2000_${DIR}/
#     cd -
# done

for DIR in event_2000_dr2_*; do
    cd ${DIR}
    echo "phi_correlation='${DIR}'"
    time python ../../../python/plot_pca.py --model model_weights.pth --tell_me_phi_correlation
    cd ../
    # break
done

