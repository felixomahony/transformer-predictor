import numpy as np
import os
import torch


class GS_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_folder,
        temporal=False,
        conditionalise_dim=-1,
        temporal_size=32,
        temporal_window=4,
        spatial_size=7,
        codebook_size=1000,
        empty_delta=2,
        train=True,
        split=0.8,
    ):
        """
        Args:
        data_folder: folder containing the data
        temporal: whether data should have a temporal dimension
        conditionalise_dim: we can further conditionalise to a single dimension (i.e. slice the data in a spatial dimension to yield a 2D slice)
        temporal_size: number of frames each scene has (i.e. maximum temporal window)
        temporal_window: number of frames to use for each sample
        spatial_size: size of the spatial cube
        codebook_size: size of the codebook
        empty_delta: the codebook index to use for empty locations, relative to the highest codebook index
        train: whether to use the training or testing set
        split: fraction of scenes to use for training
        """
        self.data_folder = data_folder
        self.temporal = temporal
        self.temporal_window = temporal_window
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size
        self.codebook_size = codebook_size
        self.empty_delta = empty_delta
        self.conditionalise_dim = None if conditionalise_dim == -1 else conditionalise_dim
        self.spatial_multiplier_ = 1 if self.conditionalise_dim is None else self.spatial_size

        n_scenes_tot = len(os.listdir(data_folder))
        n_scenes_train = int(n_scenes_tot * split)
        if train:
            self.scenes = range(n_scenes_train)
        else:
            self.scenes = range(n_scenes_train, n_scenes_tot)


    def load_sf(self, scene, frame, slce=None):
        path = os.path.join(
            self.data_folder, f"scene_{scene:04d}", f"frame_{frame:04d}.npz"
        )
        data = np.load(path)
        spatial_locations = data["spatial_locations"]
        codebook_indices = data["features"]

        n_spatial_dims = 3
        if self.conditionalise_dim is not None:
            n_spatial_dims -= 1
            if slce is None:
                raise ValueError("slce must be provided if we are conditionalising")
            if slce not in spatial_locations[:, self.conditionalise_dim]:
                slce = spatial_locations[:, self.conditionalise_dim][0]
            spatial_locations_mask = spatial_locations[:, self.conditionalise_dim] == slce
            spatial_locations = spatial_locations[spatial_locations_mask]
            codebook_indices = codebook_indices[spatial_locations_mask]
            spatial_dims = [i for i in range(3) if i != self.conditionalise_dim]
            spatial_locations = spatial_locations[:, spatial_dims]

        retval = np.full(
            self.spatial_size ** n_spatial_dims,
            self.codebook_size + self.empty_delta - 1,
        )
        spatial_indices = np.ravel_multi_index(
            spatial_locations.T,
            [self.spatial_size] * n_spatial_dims,
        )
        retval[spatial_indices] = codebook_indices
        return retval, torch.tensor(0)

    def __len__(self):
        if self.temporal:
            return len(self.scenes) * (self.temporal_size - self.temporal_window + 1) * self.spatial_multiplier_
        else:
            return len(self.scenes) * self.temporal_size * self.spatial_multiplier_
        
    def datum_size(self):
        ds = self.spatial_size ** (3 if self.conditionalise_dim is None else 2)
        if self.temporal:
            ds *= self.temporal_window
        return ds

    def __getitem__(self, idx):
        if self.temporal:
            slce = idx % self.spatial_multiplier_
            idx //= self.spatial_multiplier_
            scene = self.scenes[idx]
            retval = [self.load_sf(scene, frame, slce) for frame in range(self.temporal_window)]
            retval = tuple(np.stack([r[i] for r in retval], axis=0).flatten() for i in range(len(retval[0])))
            return retval
        else:
            slce = idx % self.spatial_multiplier_
            idx //= self.spatial_multiplier_
            scene_idx = idx // self.temporal_size
            scene = self.scenes[scene_idx]
            frame = idx % self.temporal_size
            return self.load_sf(scene, frame, slce)
        
if __name__ == "__main__":
    data_folder = "/scratch/foo22/Data/Physics_Simulation/intermediate_data/codebook/four_compression/bp_64/codebook_indices/"
    ts=32
    sps = 7
    n_scenes = int(400*0.8)

    # first test full dataset
    tw = 4
    ds_full = GS_Dataset(data_folder, temporal=True, temporal_size=ts, spatial_size=sps, temporal_window=tw, train=True)
    assert len(ds_full) == (ts-tw+1) * n_scenes, f"len(ds_full) = {len(ds_full)}, expected {(ts-tw+1) * n_scenes}"
    assert len(ds_full[0][0]) == ds_full.datum_size(), f"ds_full[0][0] = {ds_full[0][0]}, expected {ds_full.datum_size()}"
    print("Full dataset")
    print(ds_full[0][0].shape)
    print("-------")

    # without temporal component
    ds_no_temp = GS_Dataset(data_folder, temporal=False, spatial_size=sps, temporal_window=tw, train=True)
    assert len(ds_no_temp) == (ts) * n_scenes, f"len(ds_full) = {len(ds_no_temp)}, expected {(ts) * n_scenes}"
    assert len(ds_no_temp[0][0]) == ds_no_temp.datum_size(), f"ds_no_temp[0][0] = {ds_no_temp[0][0]}, expected {ds_no_temp.datum_size()}"
    print("No temporal")
    print(ds_no_temp[0][0].shape)
    print("-------")

    # taking slices
    cd = 2
    ds_slice = GS_Dataset(data_folder, temporal=False, conditionalise_dim=cd, temporal_size=ts, spatial_size=sps, train=True)
    assert len(ds_slice) == (ts) * n_scenes * sps, f"len(ds_full) = {len(ds_slice)}, expected {(ts) * n_scenes * sps}"
    assert len(ds_slice[0][0]) == ds_slice.datum_size(), f"ds_slice[0][0] = {ds_slice[0][0]}, expected {ds_slice.datum_size}"
    print("Spatial slice")
    print(ds_slice[0][0].shape)
    print("-------")