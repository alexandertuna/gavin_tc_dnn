rm -f model_weights_ptetaphi.pth
ln -s ../embed_ptetaphi_event_2000_reference/model_weights_ptetaphi.pth ./
time python ../../python/main_physics.py --load_model --load_embedding_model
