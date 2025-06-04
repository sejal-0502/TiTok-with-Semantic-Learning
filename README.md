# TiTok with Semantic Learning
Training and evaluation of 1-Dimensional tokenizer based on Vision Transformers

## Requirements
A suitable environment (For eg. `mamba`) named `tokenization` can be created and activated with:

```
mamba env create -f environment.yaml
conda activate tokenization
```

### Train TiTok from scratch
```
python main.py --base ./configs/baseline.yaml -t True --n_gpus 1 --n_nodes 1  --name baseline
```
n_gpus: specifies number of gpus, default=1 \
n_nodes: specifies number of nodes, default=1

### Fine-tune from previous checkpoint
```
python main.py --base ./configs/baseline.yaml -t True --n_gpus 1  --resume ./logs/path/to/checkpoint
```

### Compute FID score
```
python tools/compute_codebook_usage.py --config_path ./vqgan_logs/baseline_exp/config.yaml --ckpt_path ./vqgan_logs/baseline_exp/checkpoints/last.ckpt --compute_rFID_score
```
