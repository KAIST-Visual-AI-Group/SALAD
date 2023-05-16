import h5py
import numpy as np
import pandas as pd
import torch
from dotmap import DotMap

from salad.utils.paths import DATA_DIR
from salad.utils import thutil


class SALADDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__()
        self.data_path = str(DATA_DIR / data_path)
        self.repeat = repeat
        self.__dict__.update(kwargs)
        self.hparams = DotMap(self.__dict__)

        """
        Global Data statistics.
        """
        if self.hparams.get("global_normalization"):
            with h5py.File(self.data_path.replace(".hdf5", "_mean_std.hdf5")) as f:
                self.global_mean = f["mean"][:].astype(np.float32)
                self.global_std = f["std"][:].astype(np.float32)

        self.data = dict()
        with h5py.File(self.data_path) as f:
            for k in self.hparams.data_keys:
                self.data[k] = f[k][:].astype(np.float32)

                """
                global_normalization arg is for gaussians only.
                """
                if k == "g_js_affine":
                    if self.hparams.get("global_normalization") == "partial":
                        assert k == "g_js_affine"
                        if self.hparams.get("verbose"):
                            print("[*] Normalize data only for pi and eigenvalues.")
                        # 3: mu, 9: eigvec, 1: pi, 3: eigval
                        self.data[k] = self.normalize_global_static(
                            self.data[k], slice(12, None)
                        )
                    elif self.hparams.get("global_normalization") == "all":
                        assert k == "g_js_affine"
                        if self.hparams.get("verbose"):
                            print("[*] Normalize data for all elements.")
                        self.data[k] = self.normalize_global_static(
                            self.data[k], slice(None)
                        )

    def __getitem__(self, idx):
        if self.repeat is not None and self.repeat > 1:
            idx = int(idx / self.repeat)

        items = []
        for k in self.hparams.data_keys:
            data = torch.from_numpy(self.data[k][idx])
            items.append(data)

        if self.hparams.get("concat_data"):
            return torch.cat(items, -1)  # [16,528]
        if len(items) == 1:
            return items[0]
        return items

    def __len__(self):
        k = self.hparams.data_keys[0]
        if self.repeat is not None and self.repeat > 1:
            return len(self.data[k]) * self.repeat
        return len(self.data[k])

    def get_other_latents(self, key):
        with h5py.File(self.data_path) as f:
            return f[key][:].astype(np.float32)

    def normalize_global_static(self, data: np.ndarray, normalize_indices=slice(None)):
        """
        Input:
            np.ndarray or torch.Tensor. [16,16] or [B,16,16]
            slice(None) -> full
            slice(12, None) -> partial
        Output:
            [16,16] or [B,16,16]
        """
        assert normalize_indices == slice(None) or normalize_indices == slice(
            12, None
        ), print(f"{normalize_indices} is wrong.")
        data = thutil.th2np(data).copy()
        data[..., normalize_indices] = (
            data[..., normalize_indices] - self.global_mean[normalize_indices]
        ) / self.global_std[normalize_indices]
        return data

    def unnormalize_global_static(
        self, data: np.ndarray, unnormalize_indices=slice(None)
    ):
        """
        Input:
            np.ndarray or torch.Tensor. [16,16] or [B,16,16]
            slice(None) -> full
            slice(12, None) -> partial
        Output:
            [16,16] or [B,16,16]
        """
        assert unnormalize_indices == slice(None) or unnormalize_indices == slice(
            12, None
        ), print(f"{unnormalize_indices} is wrong.")
        data = thutil.th2np(data).copy()
        data[..., unnormalize_indices] = (
            data[..., unnormalize_indices]
        ) * self.global_std[unnormalize_indices] + self.global_mean[unnormalize_indices]
        return data


class LangSALADDataset(SALADDataset):
    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__(data_path, repeat, **kwargs)

        # self.game_data = pd.read_csv(self.hparams.lang_data_path)
        self.game_data = pd.read_csv(DATA_DIR / "autosdf_spaghetti_intersec_game_data.csv")
        self.shapenet_ids = np.array(self.game_data["sn"])
        self.spaghetti_indices = np.array(self.game_data["spaghetti_idx"])  # for 5401
        self.texts = np.array(self.game_data["text"])

        assert len(self.shapenet_ids) == len(self.spaghetti_indices) == len(self.texts)

    def __getitem__(self, idx):
        if self.repeat is not None and self.repeat > 1:
            idx = int(idx / self.repeat)

        spa_idx = self.spaghetti_indices[idx]
        text = self.texts[idx]
        latents = []
        for k in self.hparams.data_keys:
            data = torch.from_numpy(self.data[k][spa_idx])
            latents.append(data)

        item = latents + [text]
        if self.hparams.get("concat_data"):
            latents = torch.cat(latents, -1)
            return latents, text

        return item

    def __len__(self):
        if self.repeat is not None and self.repeat > 1:
            return len(self.texts) * self.repeat
        return len(self.texts)
