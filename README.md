<h2 align="center">
  SAR-DEIM: Real-Time and Robust SAR Ship Detection with High Accuracy  
</h2>

## 1. Quick start

### Setup

```shell
conda create -n deim python=3.11.9
conda activate deim
pip install -r requirements.txt
```


### 2.Data Preparation
## Usage
<details open>
<summary> COCO2017 </summary>

1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml --use-amp --seed=0
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml --test-only -r model.pth
```

<!-- <summary>3. Tuning </summary> -->
3. Tuning
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml --use-amp --seed=0 -t model.pth
```
</details>


THE DATASET~
https://pan.baidu.com/s/1nzyoekGwG-cxM4YufXnPfg 
Extraction code: tdjc 



✨ Feel free to contribute and reach out if you have any questions! ✨ zonghezhou5@gmail.com
