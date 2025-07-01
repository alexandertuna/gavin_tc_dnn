import numpy as np
import torch

VERY_SIMILAR = 0.005

class InspectT5T5:

    def __init__(self, processor, trainer):
        self.processor = processor
        self.trainer = trainer


    def inspect(self):
        for batch, (feats_l, feats_r) in enumerate(self.get_t5t5_nonduplicates_lowdistances()):
            if batch > 0:
                break
            # if batch < 3:
            #     continue
            # if batch > 3:
            #     break
            self.compare_t5_features(feats_l, feats_r)
        # feats_l, feats_r = self.get_t5t5_nonduplicates_lowdistances()
        # self.compare_t5_features(feats_l, feats_r)


    def get_t5t5_nonduplicates_lowdistances(self):

        self.trainer.embed_t5.eval()

        with torch.no_grad():
            for batch, (x_left, x_right, y, _) in enumerate(self.trainer.test_t5_loader):
                #if batch == 0:
                #    continue
                e_l = self.trainer.embed_t5(x_left)
                e_r = self.trainer.embed_t5(x_right)
                d   = torch.sqrt(((e_l - e_r) ** 2).sum(dim=1, keepdim=True) + 1e-6)
                target = (d < VERY_SIMILAR).flatten() & (y == 1).flatten()
                print(target.sum())
                print(x_left[target].shape)
                print(x_right[target].shape)
                print(f"Batch {batch}")
                yield x_left[target].numpy(), x_right[target].numpy()


    def compare_t5_features(self, x_l, x_r):
        nevt = len(self.processor.features_per_event)
        for i_l in range(len(x_l)):
            print("*"*60)
            for evt in range(nevt):
                features = self.processor.features_per_event[evt]
                sim_idxs = self.processor.sim_indices_per_event[evt]
                exists_l = (x_l[i_l] == features).all(axis=1)
                exists_r = (x_r[i_l] == features).all(axis=1)
                if exists_l.any() and exists_r.any():
                    print(f"Event {evt} has {len(features)} features {type(features)} -> {features.shape}")
                    print(f"Event {evt} has {len(sim_idxs)} sim_idxs {type(sim_idxs)} -> {sim_idxs.shape}")
                    idxs_l = np.argwhere(exists_l).squeeze()
                    idxs_r = np.argwhere(exists_r).squeeze()
                    sim_idxs_l = sim_idxs[idxs_l]
                    sim_idxs_r = sim_idxs[idxs_r]
                    print(f"idxs_l", idxs_l)
                    print(f"idxs_r", idxs_r)
                    print(f"feature_l", x_l[i_l])
                    print(f"feature_r", x_r[i_l])
                    print(f"equal?", (x_l[i_l] == x_r[i_l]).all())
                    print(f"sim_idxs_l", sim_idxs_l, np.unique(sim_idxs_l))
                    print(f"sim_idxs_r", sim_idxs_r, np.unique(sim_idxs_r))
                    print("")
            print("*"*60)
            # print(f"Event {evt} has {len(features)} features {type(features)} -> {features.shape}")
            # if evt > 10:
            #     break





