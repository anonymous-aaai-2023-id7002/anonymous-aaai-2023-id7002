## Dynamic Ensemble of Low-fidelity Experts：Mitigating NAS Cold-Start
--------

There are several python files and `cfgs` dir under this directory. The python files are:

* `datasets.py`: Codes for dataset construction
* `utils.py`: Utils for predictor training and validation
* `train_predictor.py`: Directly train the predictor on the actual performance data, or pretrain the predictor on a single type of low-fidelity information data and finetune on the actual performanda data
* `dynamic_ensemble_train_predictor.py`: Conduct predictor training with the proposed dynamic ensemble framework

### Data Preparation
Download the data from [here](https://drive.google.com/drive/folders/1bdqGXNioj0xp_P6TmLiEHScP3aCj4iGl?usp=sharing).

### Run Predictor Training
To run the proposed dynamic ensemble predictor training framework, use the `dynamic_ensemble_train_predictor.py` script. Specifically, run `python dynamic_ensemble_train_predictor.py <CFG_FILE> --search-space <SEARCH_SPACE> --train-ratio <TRAIN_RATIO> --pretrain-ratio <PRETRAIN_RATIO> --train-pkl <TRAIN_PKL> --valid-pkl <VALID_pkl> --seed <SEED> --gpu <GPU_ID> --train-dir <TRAIN_DIR>`, where:

* `CFG_FILE`: Path of the configuration file
* `SEARCH_SPACE`: The search space (including nasbench-201 / nb301 / nds)
* `TRAIN_RATIO`: Proportion of training samples used in the second-step training
* `PRETRAIN_RATIO`: Proportion of training sampled used in the first-step training. Default: 1.0
* `TRAIN_PKL`: Path of the training data
* `VALID_PKL`: Path of the validation data
* `SEED`: Seed (optional)
* `GPU_ID`: ID of the used GPU. Currently, we only support single-GPU training. Default: 0
* `TRAIN_DIR`: Path to save the logs and results

To run the pretrain-and-finetune predictor training flow, use the `train_predictor.py` script. Specifically, run `python train_predictor.py <CFG_FILE> --search-space <SEARCH_SPACE> --train-ratio <TRAIN_RATIO> --pretrain-ratio <PRETRAIN_RATIO> --train-pkl <TRAIN_PKL> --valid-pkl <VALID_pkl> --low-fidelity-type <LOW_FIDELITY_TYPE> --seed <SEED> --gpu <GPU_ID> --train-dir <TRAIN_DIR>`, where:

* `LOW_FIDELITY_TYPE`: Type of the utilized low-fidelity information. Default: one_shot

By set `--no-pretrain`, we can run the vanilla predictor training flow without utilizing low-fidelity information. 

We provide example predictor training configuration files under `./cfgs`, including:

* `train_nb201_gates_config.yaml`:
  * Encoder: GATES
  * Search space: nasbench-201
  * Objective: Ranking loss
  * Method: Vanilla or the pretrain & finetune method as described in Introduction.
* `dynamic_ensemble_nb201_gates_config.yaml`
  * Encoder: GATES
  * Search space: nasbench-201
  * Objective: Ranking loss (DELE only support ranking loss)
  * Method: Dynamic ensemble
  
For example, run dynamic ensemble predictor training on NAS-Bench-201 with 100% training samples by

`python dynamic_ensemble_train_predictor.py cfgs/dynamic_ensemble_nb201_gates_config.yaml --search-space nasbench-201 --train-ratio 1.0 --train-pkl data/NAS-Bench-201/nasbench201_train.pkl --valid-pkl data/NAS-Bench-201/nasbench201_valid.pkl --train-dir <TRAIN_DIR>`

Other example configuration files are uploaded [here](https://drive.google.com/drive/folders/1DqH5M3ryv6ohONtgpy8ZSbeYpsHidXI1?usp=sharing).

### Available Types of Low-fidelity Information
Available types of low-fidelity information for different search spaces include:
- NAS-Bench-201: grad_norm, snip, grasp, fisher, jacob_cov, plain, synflow, flops, params, relu, relu_logdet, one_shot, latency
- NAS-Bench-301: one_shot, grad_norm, snip, grasp, fisher, jacob_cov, plain, synflow, synflow_bn, flops, params
- NDS-ResNet / ResNeXT-A: relu_logdet, grad_norm, grasp, jacob_cov, plain, relu, synflow, one_shot, params, flops
