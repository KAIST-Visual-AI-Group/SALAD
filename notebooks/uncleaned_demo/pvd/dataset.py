import torch
import pandas as pd
import numpy as np
import h5py
from dotmap import DotMap
import jutils
#### < Seungwoo >
# Comment-out module for language condition
# from partglot.utils.simple_utils import unpickle_data
# from partglot.datamodules.data_utils import convert_labels_to_one_hot, shuffle_game_geometries, get_mask_of_game_data, pad_text_symbols_with_zeros
####


def scale_eigenvectors(gaus: torch.Tensor):
    """
    gaus: [16,16]
    """
    mu = gaus[:, :3]
    eig_vec = gaus[:, 3:12]  # [--v1-- --v2-- --v3--]
    pi = gaus[:, 12:13]
    eig_val = gaus[:, 13:]  # [16, 3]
    # print("eig_val:", eig_val)
    eig_vec = eig_vec.reshape(-1, 3, 3) * eig_val.reshape(-1, 3, 1)
    eig_vec = eig_vec.reshape(-1, 9)

    return torch.cat([mu, eig_vec, pi], 1)  # [16, 3+9+1=13]


def split_eigens(gaus: torch.Tensor):
    """
    Input: torch.Tensor(16, 13) [mu, eig_vec, pi]
    Output: torch.Tensor(16, 16) [mu, eig_vec, pi, eig_val]
    """
    mu = gaus[:, :3]
    eig_vec = gaus[:, 3:12]
    pi = gaus[:, 12:13]
    eig_vec = eig_vec.reshape(-1, 3, 3)
    """
    [ --v1-- \\
      --v2-- \\
      --v3-- ]
    """
    eig_val = eig_vec.norm(p=2, dim=-1)  # [16, 3]
    eig_vec = eig_vec / eig_val.reshape(-1, 3, 1)
    eig_vec = eig_vec.reshape(-1, 9)

    return torch.cat([mu, eig_vec, pi, eig_val], -1)


def batch_split_eigens(gaus: torch.Tensor):
    """
    Input: torch.Tensor(B,16,13)
    Output: torch.Tensor(B,16,13)
    """
    B, G = gaus.shape[:2]
    mu = gaus[..., :3]
    eig_vec = gaus[..., 3:12]
    pi = gaus[..., 12:13]

    eig_vec = eig_vec.reshape(B, G, 3, 3)
    eig_val = eig_vec.norm(p=2, dim=-1)  # [B,G,3]
    eig_vec = eig_vec / eig_val.unsqueeze(-1)
    eig_vec = eig_vec.reshape(B, G, 9)

    return torch.cat([mu, eig_vec, pi, eig_val], -1)


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, num_points, split="train", **kwargs):
        super().__init__()
        self.num_points = num_points
        self.split = split

        DATA_ROOT = "/home/juil/datasets/ShapeNet-Part-HDF5"
        with open(f"{DATA_ROOT}/{split}_hdf5_file_list.txt") as f:
            h5_path_list = [f"{DATA_ROOT}/{line.rstrip()}" for line in f]

        self.points = []
        for h5_path in h5_path_list:
            data = h5py.File(h5_path)
            label = data["label"][:].astype(np.float32).squeeze()
            pts = data["data"][label == 4].astype(np.float32)
            self.points.append(pts)

        self.points = np.concatenate(self.points, 0)

    def __getitem__(self, idx):
        pts = self.points[idx][: self.num_points]
        offset, scale = jutils.meshutil.get_offset_and_scale(pts)
        pts = (pts - offset) / scale
        return pts

    def __len__(self):
        return len(self.points)


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.repeat = repeat

        with h5py.File(data_path) as f:
            self.data = f["data"][:]

    def __getitem__(self, idx):
        if self.repeat is not None and self.repeat > 1:
            idx = int(idx / self.repeat)
        return torch.from_numpy(self.data[idx].astype(np.float32))

    def __len__(self):
        if self.repeat is not None and self.repeat > 1:
            return len(self.data) * self.repeat
        return len(self.data)


class SpaghettiLatentDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.repeat = repeat
        self.__dict__.update(kwargs)
        self.hparams = DotMap(self.__dict__)

        """
        Global Data statistics.
        """
        if self.hparams.get("global_normalization") or self.hparams.get(
            "sj_global_normalization"
        ):
            with h5py.File(self.data_path.replace(".hdf5", "_mean_std.hdf5")) as f:
                self.global_mean = f["mean"][:].astype(np.float32)
                self.global_std = f["std"][:].astype(np.float32)

                self.sj_global_mean = f["sj_mean"][:].astype(np.float32)
                self.sj_global_std = f["sj_std"][:].astype(np.float32)

        self.data = dict()
        with h5py.File(data_path) as f:
            for k in self.hparams.data_keys:
                self.data[k] = f[k][:].astype(np.float32)

                """
                global_normalization arg is for gaussians only.
                sj_global_normalization arg is for intrinsics.
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
                if k == "s_j_affine":
                    if self.hparams.get("sj_global_normalization"):
                        if k == "s_j_affine":
                            if self.hparams.get("verbose"):
                                print("[*] Normalize intrinsics.")
                            self.data[k] = self.normalize_sj_global_static(self.data[k])

    def __getitem__(self, idx):
        if self.repeat is not None and self.repeat > 1:
            idx = int(idx / self.repeat)

        items = []
        for k in self.hparams.data_keys:
            data = torch.from_numpy(self.data[k][idx])
            if self.hparams.get("scale_eigenvectors") and k == "g_js_affine":
                data = scale_eigenvectors(data)
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
        data = jutils.thutil.th2np(data).copy()
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
        data = jutils.thutil.th2np(data).copy()
        data[..., unnormalize_indices] = (
            data[..., unnormalize_indices]
        ) * self.global_std[unnormalize_indices] + self.global_mean[unnormalize_indices]
        return data

    def normalize_sj_global_static(
        self, data: np.ndarray, normalize_indices=slice(None)
    ):
        """
        Input:
            np.ndarray or torch.Tensor. [16,16] or [B,16,16]
            slice(None) -> full
            slice(12, None) -> partial
        Output:
            [16,16] or [B,16,16]
        """
        assert normalize_indices == slice(None), print(f"{normalize_indices} is wrong.")
        data = jutils.thutil.th2np(data)
        data[..., normalize_indices] = (
            data[..., normalize_indices] - self.sj_global_mean[normalize_indices]
        ) / self.sj_global_std[normalize_indices]
        return data

    def unnormalize_sj_global_static(
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
        assert unnormalize_indices == slice(None), print(
            f"{unnormalize_indices} is wrong."
        )
        data = jutils.thutil.th2np(data)
        data[..., unnormalize_indices] = (
            data[..., unnormalize_indices]
        ) * self.sj_global_std[unnormalize_indices] + self.sj_global_mean[
            unnormalize_indices
        ]
        return data


class T5SpaghettiLatentDataset(SpaghettiLatentDataset):
    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__(data_path, repeat, **kwargs)

        with h5py.File(self.hparams.lang_data_path) as f:
            self.lang_embeds = f["embed"][:][:, 0].astype(np.float32)
            self.spaghetti_ids = f["spaghetti_id"][:].astype(np.long)
            self.original_texts = f["text"][:]

        """
        Rearange SPAGHETTI index in h5 to align them with aug h5 indices.
        """
        with open(
            "/home/juil/projects/3D_CRISPR/crispr/data/pre_trained_spaghetti_latent_params/intersec_spaghetti_im_train_with_index.txt",
            "r",
        ) as f:
            self.sorted_spa_ids = [
                int(line.rstrip().split(" ")[0]) for line in f.readlines()[1:]
            ]

        # print("[*] Rearange spaghetti_ids to make it go with augmented latent data.")
        # with open("/home/juil/projects/3D_CRISPR/crispr/data/pre_trained_spaghetti_latent_params/intersec_spaghetti_im_train_with_index.txt") as f:
        # self.sorted_spaghetti_ids = [int(line.rstrip().split(" ")[0]) for line in f.readlines()[1:]]
        # # 0, 1, 2, 3, 4, 6, 7, ...
        # self.spa_ids2aug_ids = {x : i for i, x in enumerate(self.sorted_spaghetti_ids)}

        # mask = np.zeros(len(self.lang_embeds))
        # for i, idx in enumerate(self.spaghetti_ids):
        # if idx in self.sorted_spaghetti_ids:
        # mask[i] = 1
        # self.spaghetti_ids[i] = self.spa_ids2aug_ids[idx]

        # mask = mask.astype(bool)
        # self.lang_embeds = self.lang_embeds[mask]
        # self.spaghetti_ids = self.spaghetti_ids[mask]
        # self.original_texts = self.original_texts[mask]

    def __getitem__(self, idx):
        lang_embed = torch.from_numpy(self.lang_embeds[idx])
        spaghetti_idx = self.spaghetti_ids[idx]
        latents = []
        for k in self.hparams.data_keys:
            data = torch.from_numpy(self.data[k][spaghetti_idx])
            if self.hparams.get("scale_eigenvectors") and k == "g_js_affine":
                data = scale_eigenvectors(data)
            latents.append(data)

        items = latents + [lang_embed]
        return items

    def getitem_with_text(self, idx):
        items = self.__getitem__(idx)
        text = self.original_texts[idx].decode("utf-8")
        return tuple(list(items) + [text])

    def __len__(self):
        return len(self.lang_embeds)


class ShapeglotSpaghettiDataset(SpaghettiLatentDataset):
    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__(data_path, repeat, **kwargs)

        self.game_data = pd.read_csv(self.hparams.lang_data_path)
        self.shapenet_ids = np.array(self.game_data["sn"])
        self.spaghetti_indices = np.array(self.game_data["spaghetti_idx"])  # for 5401
        self.texts = np.array(self.game_data["text"])

        if self.hparams.get("only_easy_context"):
            self.game_contexts = np.array(self.game_data["context"])
            mask = self.game_contexts == "easy"
            self.shapenet_ids, self.spaghetti_indices, self.texts = list(
                map(
                    lambda x: x[mask],
                    [self.shapenet_ids, self.spaghetti_indices, self.texts],
                )
            )
        # assert self.spaghetti_indices.max() == 5400

        if self.hparams.get("max_dataset_size"):
            self.shapenet_ids, self.spaghetti_indices, self.texts = list(
                map(
                    lambda x: x[: self.hparams.max_dataset_size],
                    [self.shapenet_ids, self.spaghetti_indices, self.texts],
                )
            )
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


class ShapeglotSpaghettiDataset2(SpaghettiLatentDataset):
    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__(data_path, repeat, **kwargs)

        # with open(
            # "/home/juil/projects/3D_CRISPR/crispr/data/pre_trained_spaghetti_latent_params/intersec_spaghetti_im_train_with_index.txt",
            # "r",
        # ) as f:
            # self.sorted_spa_sn_ids = [
                # line.rstrip().split(" ")[1] for line in f.readlines()[1:]
            # ]

        with open(
            "/home/juil/projects/3D_CRISPR/crispr/data/pre_trained_spaghetti_latent_params/spaghetti_intersec_chair_ids.txt",
            "r") as f:
            self.sorted_spa_sn_ids = [line.rstrip() for line in f]

        assert len(self.sorted_spa_sn_ids) == 5401

        self.spa_sn2int = {sn : i for i, sn in enumerate(self.sorted_spa_sn_ids)}
        self.spa_int2sn = {i : sn for sn, i in self.spa_sn2int.items()}

        (
            self.game_data,
            self.word2int,
            self.int2word,
            self.int2sn,
            self.sn2int,
            self.sorted_sn,
        ) = unpickle_data("/home/juil/projects/PartGlot/data/game_data.pkl")
        
        self.geo_ids = np.array(self.game_data[['chair_a', 'chair_b', 'chair_c']], dtype=np.int32)
        self.target_indices = np.array(self.game_data['target_chair']).astype(np.int32)
        self.target_geo_ids = np.array([self.geo_ids[i][self.target_indices[i]] for i in range(len(self.geo_ids))])
        self.target_sn_ids = np.array(list(map(lambda x : self.int2sn[x], self.target_geo_ids)))
        self.original_texts = np.array([" ".join(x) for x in self.game_data['original_text']])

        mask = np.zeros((len(self.game_data)), dtype=bool)
        for i, sn in enumerate(self.target_sn_ids):
            if sn in self.sorted_spa_sn_ids:
                mask[i] = True
        
        mask2, _ = get_mask_of_game_data(
            self.game_data,
            self.word2int,
            self.hparams.get("only_correct"),
            self.hparams.get("only_easy_context"),
            50,
            False
                )
        mask = np.logical_and(mask, mask2)

        self.target_sn_ids, self.original_texts = list(map(lambda x : x[mask][:self.hparams.max_dataset_size], [self.target_sn_ids, self.original_texts]))

        assert len(self.target_sn_ids) == len(self.original_texts)

    def __getitem__(self, idx):
        if self.repeat is not None and self.repeat > 1:
            idx = int(idx / self.repeat)
    
        sn = self.target_sn_ids[idx]
        spa_idx = self.spa_sn2int[sn]
        text = self.original_texts[idx]
        latents = []
        for k in self.hparams.data_keys:
            data = torch.from_numpy(self.data[k][spa_idx])
            latents.append(data)

        item = latents + [text]
        if self.hparams.get("concat_data"):
            latents = torch.cat(latents, - 1)
            return latents, text
        return item
    
    def __len__(self):
        if self.repeat is not None and self.repeat > 1:
            return len(self.original_texts) * self.repeat
        return len(self.original_texts)

if __name__ == "__main__":
    ds = PointCloudDataset(2048, "train")
    print(ds.__len__())
    pts = ds[0]
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1])
    plt.savefig("hi.png")
