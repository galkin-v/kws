from typing import Tuple

import hydra
import pytorch_lightning as pl
import torch
import thop
from torchmetrics import Accuracy


class KWS(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()

        self.conf = conf

        self.model = hydra.utils.instantiate(conf.model)
        self.train_acc = Accuracy(
            task="multiclass", num_classes=conf.model.n_classes, top_k=1
        )
        self.valid_acc = Accuracy(
            task="multiclass", num_classes=conf.model.n_classes, top_k=1
        )

        self.loss = torch.nn.NLLLoss()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logprobs = self.model(inputs)
        preds = logprobs.argmax(1)
        return logprobs, preds

    def on_train_start(self):

        features_params = self.conf.train_dataloader.dataset.transforms[0]

        sample_inputs = torch.randn(
            1,
            features_params.n_mels,
            features_params.sample_rate // features_params.hop_length + 1,
            device=self.device,
        )
        # don't pre-create thop buffers ('total_ops'/'total_params') - thop
        # registers them internally via register_buffer. Leave modules untouched.

        try:
            macs, params = thop.profile(self.model, inputs=(sample_inputs,))
            self.log("MACs", macs)
            self.log("Params", params)
        except Exception as e:
            # Profiling is optional -- don't fail training if thop fails (e.g., unsupported ops)
            self.log("MACs", -1)
            self.log("Params", -1)
            # avoid using self.print() here because the module may not be attached to a Trainer
            print(f"Warning: thop profiling failed: {e}")

        # Ensure modules have thop buffers that hooks expect during forward
        for m in self.model.modules():
            # register buffers only if missing to avoid KeyError
            if "total_ops" not in m._buffers:
                m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
            if "total_params" not in m._buffers:
                try:
                    total = sum(p.numel() for p in m.parameters())
                    # register a 1-dim tensor so thop hooks can index [0]
                    m.register_buffer(
                        "total_params",
                        torch.tensor([total], dtype=torch.int64),
                    )
                except Exception:
                    m.register_buffer("total_params", torch.zeros(1, dtype=torch.int64))

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch

        logprobs, preds = self.forward(inputs)

        loss = self.loss(logprobs, labels)

        log = {
            "train/loss": loss,
            "lr": self.optimizers().param_groups[0]["lr"],
            "train/accuracy": self.train_acc(preds, labels),
        }

        self.log_dict(log, on_step=True)

        return {"loss": loss}

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch

        logprobs, preds = self.forward(inputs)

        loss = self.loss(logprobs, labels)
        self.valid_acc.update(preds, labels)

        return {"loss": loss}

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        ids, inputs, _ = batch

        logprobs, preds = self.forward(inputs)
        # return probabilities (CPU) so downstream code can apply corrections/calibration
        probs = logprobs.exp().detach().cpu()
        return ids, probs

    def on_validation_epoch_end(self):
        self.log("val/accuracy", self.valid_acc.compute())
        self.valid_acc.reset()

    def train_dataloader(self):
        return hydra.utils.instantiate(self.conf.train_dataloader)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.conf.val_dataloader)

    def predict_dataloader(self):
        return hydra.utils.instantiate(self.conf.predict_dataloader)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.conf.optim,
            params=self.model.parameters(),
        )
        return {"optimizer": optimizer}
