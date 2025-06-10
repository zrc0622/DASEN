# DASEN

## 训练

```bash
# UPDeT (baseline)
python3 src/main.py --total-config=default --alg-config=qmix --env-config=sc2 --seed=354881840 with env_args.map_name=5m_vs_6m

# DASEN-v1
python3 src/main.py --total-config=dasen --agent=dasen_v1 --alg-config=qmix --env-config=sc2 --seed=612444058 with env_args.map_name=5m_vs_6m

# DASEN-v2
python3 src/main.py --total-config=dasen --agent=dasen_v2 --alg-config=qmix --env-config=sc2 --seed=603346636 with env_args.map_name=5m_vs_6m

# DASEN-v3
python3 src/main.py --total-config=dasen --agent=dasen_v3 --alg-config=qmix --env-config=sc2 --seed=869420052 with env_args.map_name=5m_vs_6m

# DASEN-v3 + SE-QMIX
python3 src/main.py --total-config=dasen --agent=dasen_v3 --alg-config=se_qmix --env-config=sc2 --seed=918940350 with env_args.map_name=5m_vs_6m
```

## 零样本泛化

```bash
# UPDeT (baseline)
python3 src/main.py --total-config=default_zero --alg-config=qmix --env-config=sc2_zero with env_args.map_name=8m_vs_9m

# DASEN-v1
python3 src/main.py --total-config=dasen_zero --agent=dasen_v1 --alg-config=qmix --env-config=sc2_zero with env_args.map_name=8m_vs_9m

# DASEN-v2
python3 src/main.py --total-config=dasen_zero --agent=dasen_v2 --alg-config=qmix --env-config=sc2_zero with env_args.map_name=8m_vs_9m

# DASEN-v3
python3 src/main.py --total-config=dasen_zero --agent=dasen_v3 --alg-config=qmix --env-config=sc2_zero with env_args.map_name=8m_vs_9m

# DASEN-v3 + SE-QMIX
python3 src/main.py --total-config=dasen_zero --agent=dasen_v3 --alg-config=se_qmix --env-config=sc2_zero with env_args.map_name=8m_vs_9m
```

## 实验结果

|             | **UPDeT**     | **DASEN-v1**        | **DASEN-v2**    | **DASEN-v3**       | **DASEN-v3+SE-QMIX** |
|-------      |-------        |-------              |-------          |--------            |--------              |
| 8m          | 77.1±4.4      | 89.8±3.2            | 97.4±1.6        | 90.9±3.1           |  96.1±1.3            |
| 8m_vs_9m    | 14.3±2.4      | 20.9±3.4            | 24.5±3.2        | 48.7±4.9           |  62.3±5.5            |
| 10m_vs_11m  | 1.9±1.6       | 2.8±1.4             | 17.6±3.2        | 27.6±3.6           |  54.5±3.7            |

- UPDeT：0425003409-updet-qmix-5m_vs_6m-seed-354881840
- DASEN-v1：0425003554-dasen_v1-qmix-5m_vs_6m-skill-4-seed-612444058
- DASEN-v2：0430235226-dasen_v2-qmix-5m_vs_6m-skill-4-seed-603346636
- DASEN-v3：0423210303-dasen_v3-qmix-5m_vs_6m-skill-4-seed-869420052
- DASEN-v3+SE-QMIX：0425233952-dasen_v3-se_qmix-5m_vs_6m-skill-4-seed-918940350

<!-- 
Python 3.7.16
 -->

<!-- # UPDeT
Official Implementation of [UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers](https://openreview.net/forum?id=v9c7hr9ADKx) (ICLR 2021 spotlight)

The framework is inherited from [PyMARL](https://github.com/oxwhirl/pymarl). [UPDeT](https://github.com/hhhusiyi-monash/UPDeT) is written in [pytorch](https://pytorch.org) and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Installation instructions

#### Installing dependencies:

```shell
pip install -r requirements.txt
```

#### Download SC2 into the `3rdparty/` folder and copy the maps necessary to run over. 

```shell
bash install_sc2.sh
```


## Run an experiment 

Before training your own transformer-based multi-agent model, there are a list of things to note.

- Currently, this repository supports marine-based battle scenarios. e.g. `3m`, `8m`, `5m_vs_6m`. 
- If you are interested in training a different unit type, carefully modify the ` Transformer Parameters` block at  `src/config/default.yaml` and revise the `_build_input_transformer` function in `basic_controller.python`.
- Before running the experiment, check the agent type in ` Agent Parameters` block at `src/config/default.yaml`.
- This repository contains two new transformer-based agents from the [UPDeT paper](https://arxiv.org/pdf/2101.08001.pdf) including 
   - Standard UPDeT
   - Aggregation Transformer

#### Training script 

```shell
python3 src/main.py --config=vdn --env-config=sc2 with env_args.map_name=5m_vs_6m
```
All results will be stored in the `Results/` folder.

## Performance

#### Single battle scenario
Surpass the GRU baseline on hard `5m_vs_6m` with:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)

![](https://github.com/hhhusiyi-monash/UPDeT/blob/main/single.png)

#### Multiple battle scenarios

Zero-shot generalize to different tasks:

- Result on `7m-5m-3m` transfer learning.

![](https://github.com/hhhusiyi-monash/UPDeT/blob/main/multi.png)

**Note: Only** UPDeT can be deployed to other scenarios without changing the model's architecture.

**More details please refer to [UPDeT paper](https://arxiv.org/pdf/2101.08001.pdf).**

## Bibtex

```tex
@article{hu2021updet,
  title={UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers},
  author={Hu, Siyi and Zhu, Fengda and Chang, Xiaojun and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2101.08001},
  year={2021}
}
```

## License

The MIT License -->