for DR2 in 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do

    DIR="dr2_norm_${DR2}"
    mkdir -p ${DIR}
    cd ${DIR}
    time python ../../../python/main.py --delta_r2_cut ${DR2}
    cd -

    DIR="dr2_proj_${DR2}"
    mkdir -p ${DIR}
    cd ${DIR}
    time python ../../../python/main.py --delta_r2_cut ${DR2} --use_phi_projection
    cd -

done

