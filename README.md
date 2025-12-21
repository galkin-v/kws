# Keyword Spotting

[Kaggle Competition](https://www.kaggle.com/competitions/keyword-spotting-mipt-2025)

constraints:
* $\le 10^4$ params
* $\le 10^6$ multiply-accumulate operations per 1 second

**How to beign**
- run `uv sync`
- hf download galkinv42/sber-keyword-spotting --repo-type dataset --local-dir data && unzip data/keyword-spotting-mipt-2023.zip -d /data

# Models

## Baseline

```bash
cd kws
python3 -m venv kws_venv
. kws_venv/bin/activate
pip install -r requirements.txt
# train
python run.py ++train_dataloader.dataset.manifest_path=<train_manifest> ++val_dataloader.dataset.manifest_path=<val_manifest> ++predict_dataloader.dataset.manifest_path=<test_manifest>
# monitor learning curves in tensorboard
tensorboard --logdir ./lightning_logs
# export to ONNX
python to_onnx.py ++init_weights=<path_to_model>
# visualize graph with netron
netron ./data/kws.onnx
# form submit for Kaggle
python submit.py ++init_weights=<path_to_model>
```

## BC-ResNet

There is an alternative BC-ResNet config inspired by "Broadcasted Residual Learning for Efficient Keyword Spotting".

To train or run with BC-ResNet, use the `conf/bcresnet.yaml` config. Example:

```bash
python run.py --config-name bcresnet
# or with overrides
python run.py --config-name bcresnet ++model.width_mult=1.5
```