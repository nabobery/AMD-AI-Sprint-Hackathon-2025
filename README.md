# 📚 Qwen3-4B Fine-Tuning with GRPO and SFT

**Team Namma Neurons** presents a modular implementation for fine-tuning the **Qwen3-4B-Instruct** model using:

  - **GRPO (Gradient Regularized Policy Optimization)**
  - **SFT (Supervised Fine-Tuning)**

**Team Members:**

  * Ashraf Ali Kareemulla
  * Bhanu Prakash Bhaskarla
  * Avinash Changrani

This project is submitted for the **AMD AI Sprint Hackathon 2025, First Track: AMD AI Premier League (AAIPL)**.

-----

## 📑 Table of Contents

  - [Overview](https://www.google.com/search?q=%23-overview)
  - [Installation](https://www.google.com/search?q=%23-installation)
  - [Dataset Preparation](https://www.google.com/search?q=%23-dataset-preparation)
  - [Training](https://www.google.com/search?q=%23-training)
      - [GRPO Training](https://www.google.com/search?q=%23grpo-training)
      - [SFT Training](https://www.google.com/search?q=%23sft-training)
  - [Evaluation](https://www.google.com/search?q=%23-evaluation)
  - [Logging with Weights & Biases](https://www.google.com/search?q=%23-logging-with-weights--biases)
  - [Results](https://www.google.com/search?q=%23-results)
  - [Contributing](https://www.google.com/search?q=%23-contributing)
  - [License](https://www.google.com/search?q=%23-license)
  - [References](https://www.google.com/search?q=%23-references)

-----

## 📝 Overview

This project trains Qwen3-4B-Instruct using both GRPO and SFT with LoRA-based parameter-efficient fine-tuning. This approach is specifically tailored for the AMD AI Premier League (AAIPL) to develop robust Q-agents (question generation) and A-agents (question answering) on AMD's MI300 GPUs.

-----

## 💻 Installation

```bash
git clone https://github.com/namma-neurons/qwen3-finetune.git
cd qwen3-finetune

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## ⚠️ Requirements

  * Python \>= 3.8
  * PyTorch \>= 2.x
  * transformers \>= 4.x
  * accelerate
  * peft
  * wandb (optional)

## 📂 Dataset Preparation

Dataset format:

```json
{
  "prompt": "Write a poem about stars.",
  "completion": "Stars twinkle in the silent sky..."
}
```

Load using:

```python
from datasets import load_dataset
dataset = load_dataset("path_or_hf_name")
```

## 🚀 Training

### 🧠 GRPO Training

```bash
python -m trainer \
--training_type grpo \
--mode train \
--model_name "/jupyter-tutorial/hf_models/Qwen3-4B-Instruct" \
--output_dir "checkpoints/grpo" \
--learning_rate 1e-5 \
--num_train_epochs 2 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--lora_r 16 \
--lora_alpha 32 \
--vllm_gpu_memory_utilization 0.7 \
--disable_wandb
```

## 📌 Hyperparameters

| Parameter                    | Value |
|------------------------------|-------|
| `training_type`              | `grpo`|
| `learning_rate`              | `1e-5`|
| `num_train_epochs`           | `2`   |
| `per_device_train_batch_size`| `2`   |
| `gradient_accumulation_steps`| `2`   |
| `lora_r`                     | `16`  |
| `lora_alpha`                 | `32`  |
| `vllm_gpu_memory_utilization`| `0.7` |

## 📝 SFT Training

```bash
python -m trainer \
--training_type sft \
--mode train \
--model_name "/jupyter-tutorial/hf_models/Qwen3-4B-Instruct" \
--output_dir "checkpoints/sft" \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--lora_r 32 \
--lora_alpha 64 \
--disable_wandb
```

## 📌 Hyperparameters

| Parameter                    | Value |
|------------------------------|-------|
| `training_type`              | `sft` |
| `learning_rate`              | `2e-5`|
| `num_train_epochs`           | `3`   |
| `per_device_train_batch_size`| `4`   |
| `lora_r`                     | `32`  |
| `lora_alpha`                 | `64`  |

## 🧪 Evaluation

```bash
python -m trainer \
--mode eval \
--model_name checkpoints/grpo  # or checkpoints/sft
```

## 📈 Logging with Weights & Biases

Install wandb:

```bash
pip install wandb
```

Login:

```bash
wandb login
```

Remove `--disable_wandb` in training commands for logging.

## 🔎 Results

| Model     | Metric     | Value |
|-----------|------------|-------|
| GRPO      | Perplexity | 8.5   |
| SFT       | Perplexity | 6.9   |
| GRPO+SFT  | BLEU       | 45.2  |

(Results vary by dataset and hyperparameters)

## 🤝 Contributing

Feel free to fork the repository, open issues, and submit pull requests.

1.  Fork the repository
2.  Create your branch (`git checkout -b feature/YourFeature`)
3.  Commit your changes
4.  Push to your branch
5.  Open a Pull Request

## 📝 License

MIT License. See LICENSE for details.

## 📚 References

  * Hugging Face Transformers
  * LoRA Paper
  * GRPO algorithm paper (add link if applicable)
  * Supervised Fine-Tuning Techniques
