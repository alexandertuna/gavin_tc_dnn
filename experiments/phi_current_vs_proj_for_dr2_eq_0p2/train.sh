time python ../../python/main.py --delta_r2_cut 0.2
mv pairs_t5t5.pkl pairs_t5t5.norm.pkl
mv pairs_t5pls.pkl pairs_t5pls.norm.pkl
mv pairs_plspls.pkl pairs_plspls.norm.pkl
mv model_weights.pth model_weights.norm.pth
mv plots.pdf plots.norm.pdf

time python ../../python/main.py --delta_r2_cut 0.2 --use_phi_projection
mv pairs_t5t5.pkl pairs_t5t5.proj.pkl
mv pairs_t5pls.pkl pairs_t5pls.proj.pkl
mv pairs_plspls.pkl pairs_plspls.proj.pkl
mv model_weights.pth model_weights.proj.pth
mv plots.pdf plots.proj.pdf

time python ../../python/plot_pca.py \
     --pdf pca.norm.pdf \
     --pairs_t5t5 pairs_t5t5.norm.pkl \
     --pairs_t5pls pairs_t5pls.norm.pkl \
     --pairs_plspls pairs_plspls.norm.pkl \
     --model model_weights.norm.pth

time python ../../python/plot_pca.py \
     --pdf pca.proj.pdf \
     --pairs_t5t5 pairs_t5t5.proj.pkl \
     --pairs_t5pls pairs_t5pls.proj.pkl \
     --pairs_plspls pairs_plspls.proj.pkl \
     --model model_weights.proj.pth

