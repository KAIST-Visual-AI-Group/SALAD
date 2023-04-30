import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from salad.data.dataset import SALADDataset
from salad.utils.train_util import PolyDecayScheduler


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        network,
        variance_schedule,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = network
        self.var_sched = variance_schedule

    def forward(self, x):
        return self.get_loss(x)

    def step(self, x, stage: str):
        loss = self(x)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        x = batch
        return self.step(x, "train")

    def add_noise(self, x, t):
        """
        Input:
            x: [B,D] or [B,G,D]
            t: list of size B
        Output:
            x_noisy: [B,D]
            beta: [B]
            e_rand: [B,D]
        """
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1)  # [B,1]
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1)

        e_rand = torch.randn_like(x)
        if e_rand.dim() == 3:
            c0 = c0.unsqueeze(1)
            c1 = c1.unsqueeze(1)

        x_noisy = c0 * x + c1 * e_rand

        return x_noisy, beta, e_rand

    def get_loss(
        self,
        x0,
        t=None,
        noisy_in=False,
        beta_in=None,
        e_rand_in=None,
    ):
        if x0.dim() == 2:
            B, D = x0.shape
        else:
            B, G, D = x0.shape
        if not noisy_in:
            if t is None:
                t = self.var_sched.uniform_sample_t(B)
            x_noisy, beta, e_rand = self.add_noise(x0, t)
        else:
            x_noisy = x0
            beta = beta_in
            e_rand = e_rand_in

        e_theta = self.net(x_noisy, beta=beta)
        loss = F.mse_loss(e_theta.flatten(), e_rand.flatten(), reduction="mean")
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size=0,
        return_traj=False,
    ):
        raise NotImplementedError

    def validation_epoch_end(self, outputs):
        if self.hparams.no_run_validation:
            return
        if not self.trainer.sanity_checking:
            if (self.current_epoch) % self.hparams.validation_step == 0:
                self.validation()

    def _build_dataset(self, stage):
        if hasattr(self, f"data_{stage}"):
            return getattr(self, f"data_{stage}")
        if stage == "train":
            ds = SALADDataset(**self.hparams.dataset_kwargs)
        else:
            dataset_kwargs = self.hparams.dataset_kwargs.copy()
            dataset_kwargs["repeat"] = 1
            ds = SALADDataset(**dataset_kwargs)
        setattr(self, f"data_{stage}", ds)
        return ds

    def _build_dataloader(self, stage):
        try:
            ds = getattr(self, f"data_{stage}")
        except:
            ds = self._build_dataset(stage)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=stage == "train",
            drop_last=stage == "train",
            num_workers=4,
        )

    def train_dataloader(self):
        return self._build_dataloader("train")

    def val_dataloader(self):
        return self._build_dataloader("val")

    def test_dataloader(self):
        return self._build_dataloader("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = PolyDecayScheduler(optimizer, self.hparams.lr, power=0.999)
        return [optimizer], [scheduler]

    #TODO move get_wandb_logger to logutil.py
    def get_wandb_logger(self):
        for logger in self.logger:
            if isinstance(logger, pl.loggers.wandb.WandbLogger):
                return logger
        return None
