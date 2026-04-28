# 🏦 Banking Intent Detection - Final Project

This repository contains the complete, modular source code for fine-tuning a Large Language Model (LLM) on the **BANKING77** dataset using **Unsloth** for intent classification.

**Model:** LLaMA-3-8B (4-bit quantized)  
**Task:** Intent Classification (77 intents)  
**Accuracy on Test Set:** **88.28%**  

---

##  Guide 

This project has been structured to follow industry best practices, separating preprocessing, training, and inference into modular scripts. The configurations are centralized in YAML files. 

The standalone `inference.py` script is highly robust: **it auto-detects your hardware**. You can test this project on a high-end GPU (using Unsloth), a standard GPU (using standard Transformers + PEFT), or even on a laptop without a GPU (CPU mode).

### Step 1: Clone and Setup

```bash
# 1. Clone the repository
git clone https://github.com/Dangkhoaa28/banking-intent-unsloth.git
cd banking-intent-unsloth

# 2. Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Model Weights

Since the LoRA adapters (~160MB) may not be included directly in the repo due to size limits, please download them and place them in the `outputs/` folder.

**[Download Model Checkpoint (outputs.zip) Here](https://drive.google.com/drive/u/0/folders/1GbxQhmK-SZ8RgNzt1ImJJEuHpqCVPwGJ)**

Ensure the folder structure looks exactly like this:
```text
banking-intent-unsloth/
|-- outputs/
|   |-- adapter_config.json
|   |-- adapter_model.safetensors
|   |-- tokenizer.json
|   |-- tokenizer_config.json
|   |-- special_tokens_map.json
```

### Step 3: Run Inference (Testing the Model)

You can easily test the model using the provided bash script or python file. 
*(Note: On the very first run, it will automatically download the base LLaMA-3-8B model from HuggingFace).*

**Method 1: Using the Bash Script**
```bash
bash inference.sh "I lost my credit card, how can I freeze it?"
```

**Method 2: Using Python directly**
```bash
python scripts/inference.py "Why was I charged a fee for my card payment?"
```

**Method 3: Evaluate on the Full Test Set**
If you want to verify the reported **88.28% accuracy**, simply run:
```bash
python scripts/evaluate.py
```

*(You can also use `--max-samples 100` to do a quick test run).*

---

##  Project Structure

```text
banking-intent-unsloth
|-- scripts/
|   |-- train.py              # Loads config, data, and fine-tunes LLaMA-3
|   |-- inference.py          # Standalone IntentClassification class (Hardware auto-detect)
|   |-- evaluate.py           # Evaluates model on test.csv
|   |-- preprocess_data.py    # Downloads and samples BANKING77 dataset
|
|-- configs/
|   |-- train.yaml            # Hyperparameters for training
|   |-- inference.yaml        # Inference configurations
|
|-- sample_data/
|   |-- train.csv             # 50% stratified sample of train data
|   |-- test.csv              # Full test data
|   |-- test_results.csv      # Detailed predictions after evaluation
|
|-- outputs/                  # Saved LoRA adapters (downloaded separately)
|-- train.sh                  # Automation script for data + training
|-- inference.sh              # Automation script for inference
|-- requirements.txt          # Python dependencies
|-- banking-intent-finetuning.ipynb # Original Kaggle Notebook
```

---

##  Hyperparameters Used (train.yaml)

To ensure reproducibility, all hyperparameters exactly match the Kaggle notebook:

| Parameter | Value |
|---|---|
| Base Model | `unsloth/llama-3-8b-bnb-4bit` |
| LoRA (r / alpha / dropout) | 16 / 32 / 0 |
| Learning Rate | 2e-4 |
| Batch Size | 4 (per device) |
| Gradient Accumulation | 8 (effective batch = 32) |
| Epochs | 1 |
| Optimizer | AdamW 8-bit |
| Seed | 3407 |

---

##  Video Demonstration

The video below demonstrates how the inference script is executed, displays predictions for sample inputs, and shows the final accuracy obtained on the test set.

 **[Watch Demo Video Here](https://drive.google.com/drive/u/0/folders/1GbxQhmK-SZ8RgNzt1ImJJEuHpqCVPwGJ)**

---

**Course:** Applications of Natural Language Processing in Industry  
**Lecturer:** Dr. Nguyen Hong Buu Long  
**Institution:** University of Science - VNUHCM  
**Student:** [Nguyen Dang Khoa] - [23120134]
