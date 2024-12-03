# 1k_cnn_layers
Train a 1000-layer network to check why we can train such deep network.

The idea is to check why we even can train a 1000-layer network without any issue. Is that about Activation-functions, Normalisation Layers or Skip-Connections?

# Usage
1. Install requirements
```pip install -r req.txt```
2. If you want to use WanDB, login via `wandb login`
3. Setup your configuration in [configs/convnext.yaml](configs/convnext.yaml)
- focus on `model_blocks` as this leads to total number of layers. Each module has 3 CNN layers, so having 3-3-27-3 leads to (3*3 + 27)*3 = 108 layers. + 2 layers (first and last layer), in total 110.
- change of activation function, normalisation or skip-connection are code inside the code [src/convnext_v2.py](src/convnext_v2.py)
4. Run training via `python train.py configs/convnext.py`


