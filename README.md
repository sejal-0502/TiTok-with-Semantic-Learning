# token_compression
Training and evaluation of vector quantized tokenizer based on VQGAN

## Requirements
A suitable environment (For eg. `mamba`) named `tokenization` can be created and activated with:

```
mamba env create -f environment.yaml
conda activate tokenization
```

### Train VQGAN from scratch
```
python main.py --base ./configs/vqgan_baseline.yaml -t True --n_gpus 1 --n_nodes 1  --name vqgan123
```
n_gpus: specifies number of gpus, default=1 \
n_nodes: specifies number of nodes, default=1

### Fine-tune from previous checkpoint
```
python main.py --base ./configs/vqgan_baseline.yaml -t True --n_gpus 1  --resume ./logs/path/to/checkpoint
```