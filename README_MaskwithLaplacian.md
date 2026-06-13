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
python train_rdnet.py --name rdnet_
```

### Stage 2：ERRNet + RDNet + Simple Gate 联合训练

```bash
python train_errnet_unaligned.py --name errnet_gate --hyper -r \
    --rdnet_path checkpoints/rdnet_/latest_net_G.pth \
    --use_rdnet --rdnet_guidance gate --gate_type simple
```

> `train_errnet_unaligned.py` 中 `opt.gate_type` 需确认为 `'simple'`。

---

## 测试

### 单数据集

```bash
python test_errnet.py --name test --dataset <name> -r \
    --icnn_path checkpoints/errnet_simple_gate/latest_net_G.pth \
    --use_rdnet --rdnet_guidance gate --gate_type simple --hyper
```

可选 `<name>`：`ceilnet_table2` `real20` `postcard` `objects` `wild` `sir2_withgt`

### 自定义图片

```bash
python test_errnet.py --name test --dataset custom --input_dir <path> -r \
    --icnn_path checkpoints/errnet_simple_gate/latest_net_G.pth \
    --use_rdnet --rdnet_guidance gate --gate_type simple --hyper
```

### 批量测试

修改 [test.sh](test.sh) 中变量后运行 `bash test.sh`：

```bash
CKPT_ERRNET="checkpoints/errnet_simple_gate/latest_net_G.pth"
GATE_TYPE="simple"
USE_HYPER=1
RDNET_GUIDANCE="gate"
```
