# 📘 Quick Start

You can find our complete prompts in `verl/prompt_config.py`.

## ⚙️ 1. Environment Configuration
```bash
# step1: create conda env
conda create -n verl python==3.12

# step2: activate env and setup verl
# ref: https://verl.readthedocs.io/en/latest/start/install.html#install-dependencies
conda init
source ~/.bashrc
conda activate verl
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

## 🖥️ 2. Launch the API Service

Start the API service for model inference used in downstream recommendation score prediction.
```bash
bash scripts/run_api_call_server.sh
```
Before running the script, please make sure the model path in the script is correctly configured.

## 🏃 3. Start Training

Run the following script to launch training:

```bash
bash scripts/train.sh
```
Before running the script, please make sure the work directory, model path, dataset path, wandb settings, and logging directory are correctly configured.