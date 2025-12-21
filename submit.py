import os
from typing import List, Tuple

import hydra
import omegaconf
import pandas as pd
import torch

from src.module import KWS
from utils import omegaconf_extension


@omegaconf_extension
@hydra.main(version_base="1.2", config_path="conf", config_name="bcresnet.yaml")
def main(conf: omegaconf.DictConfig) -> None:

    os.chdir(hydra.utils.get_original_cwd())

    module = KWS(conf)
    if conf.init_weights:
        ckpt = torch.load(conf.init_weights, map_location="cpu", weights_only=False)
        module.load_state_dict(
            {k: v for k, v in ckpt["state_dict"].items() if "total" not in k}
        )

    logger = hydra.utils.instantiate(conf.logger)
    trainer = hydra.utils.instantiate(conf.trainer, logger=logger)

    predictions: List[Tuple[torch.Tensor, torch.Tensor]] = trainer.predict(
        module, return_predictions=True
    )

    ids_tensor, second_tensor = zip(*predictions)
    ids = torch.cat(ids_tensor).numpy()

    # second_tensor might be predicted labels (1D) or probability vectors (2D)
    second_cat = torch.cat(second_tensor)
    if second_cat.ndim == 1 or (second_cat.ndim == 2 and second_cat.shape[1] == 1):
        # predicted labels
        labels = second_cat.numpy().tolist()
    else:
        # probabilities
        probs = second_cat.numpy()  # shape (N, C)

        def compute_prior(manifest_path: str, idx_to_keyword: List[str]):
            df = pd.read_csv(manifest_path)
            counts = df.label.value_counts() if 'label' in df.columns else None
            if counts is None:
                # no labels in manifest -> fall back to uniform
                return pd.Series([1.0 / len(idx_to_keyword)] * len(idx_to_keyword), index=idx_to_keyword)
            # ensure index order matches idx_to_keyword
            arr = [counts.get(k, 0) for k in idx_to_keyword]
            s = pd.Series(arr, index=idx_to_keyword)
            return s / s.sum()

        def adjust_probs_with_priors(probs_arr, train_prior, target_prior, alpha=1.0, eps=1e-6):
            # apply exponent alpha to control strength of correction
            ratio = ( (target_prior + eps) / (train_prior + eps) ) ** float(alpha)
            adjusted = probs_arr * ratio[np.newaxis, :]
            adjusted /= adjusted.sum(axis=1, keepdims=True)
            return adjusted

        import numpy as np

        if conf.get('apply_prior_correction', False):
            # compute priors
            train_prior = compute_prior(conf.train_dataloader.dataset.manifest_path, conf.train_dataloader.dataset.idx_to_keyword)
            if conf.get('prior_source', 'val') == 'val':
                target_manifest = conf.val_dataloader.dataset.manifest_path
            else:
                target_manifest = conf.predict_dataloader.dataset.manifest_path
            target_prior = compute_prior(target_manifest, conf.predict_dataloader.dataset.idx_to_keyword)
            train_prior_arr = train_prior.reindex(conf.predict_dataloader.dataset.idx_to_keyword).fillna(0).values
            target_prior_arr = target_prior.reindex(conf.predict_dataloader.dataset.idx_to_keyword).fillna(0).values
            prior_alpha = float(conf.get('prior_alpha', 1.0))
            probs = adjust_probs_with_priors(probs, train_prior_arr, target_prior_arr, alpha=prior_alpha)

        labels = probs.argmax(axis=1).tolist()

    df = pd.read_csv(conf.predict_dataloader.dataset.manifest_path).iloc[ids]
    df["label"] = [
        conf.predict_dataloader.dataset.idx_to_keyword[label] for label in labels
    ]
    # ensure 'index' exists
    if "index" not in df.columns:
        df = df.reset_index()  # moves the index into a column named 'index'

    # ensure there's a 'label' column (try common alternatives or fall back to last column)
    if "label" not in df.columns:
        if "pred" in df.columns:
            df = df.rename(columns={"pred": "label"})
        elif "prediction" in df.columns:
            df = df.rename(columns={"prediction": "label"})
        elif len(df.columns) >= 2:
            # assume last column contains labels
            df = df.rename(columns={df.columns[-1]: "label"})
        else:
            raise KeyError(f"label column not found; available columns: {list(df.columns)}")

    df[["index", "label"]].to_csv("submit.csv", index=False)


if __name__ == "__main__":
    main()
