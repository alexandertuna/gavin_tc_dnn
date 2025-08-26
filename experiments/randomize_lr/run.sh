time python ../../python/main.py --num_epochs 1 --load_features --load_pairs \
--pairs_plspls pairs_plspls.norm.pkl \
--pairs_t5pls pairs_t5pls.norm.pkl \
--pairs_t5t5 pairs_t5t5.norm.pkl \
--features_pls features_pls.norm.pkl \
--features_t5 features_t5.norm.pkl \
--dont_swap_lr

time python ../../python/main.py --num_epochs 1 --load_features --load_pairs \
--pairs_plspls pairs_plspls.swap.pkl \
--pairs_t5pls pairs_t5pls.swap.pkl \
--pairs_t5t5 pairs_t5t5.swap.pkl \
--features_pls features_pls.swap.pkl \
--features_t5 features_t5.swap.pkl

