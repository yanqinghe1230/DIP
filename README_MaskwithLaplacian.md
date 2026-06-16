# 训练与测试说明

## 环境准备

```bash
pip install -r requirements.txt
python datasets/prepare_test_data.py
python datasets/prepare_train_data.py
```

## 训练

### Stage 1：RDNet 预训练

```bash
python train_rdnet.py --name rdnet_gradmask
```

### Stage 2：ERRNet + RDNet联合训练

```bash
python train_errnet.py --use_rdnet --rdnet_path <rdnet_ckpt> --rdnet_guidance gate --gate_type simple --name errnet_gate --hyper
```

## 测试

### 单数据集

```bash
python test_errnet.py --name test --dataset <name> -r \
    --icnn_path <model_ckpt> \
    --use_rdnet --rdnet_guidance gate --gate_type simple --hyper
```

可选 `<name>`：`ceilnet_table2` `real20` `postcard` `objects` `wild` 

### 批量测试

修改 [test.sh](test.sh) 中变量后运行 `bash test.sh`：

```bash
CKPT_ERRNET="checkpoints/errnet_simple_gate/latest_net_G.pth"
GATE_TYPE="simple"
USE_HYPER=1
RDNET_GUIDANCE="gate"
```
