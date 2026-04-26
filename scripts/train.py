import os
import yaml
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments

def train():
    # Load config
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['model']['model_name'],
        max_seq_length = config['model']['max_seq_length'],
        load_in_4bit = config['model']['load_in_4bit'],
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['lora']['r'],
        target_modules = config['lora']['target_modules'],
        lora_alpha = config['lora']['alpha'],
        lora_dropout = config['lora']['dropout'],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = config['training']['seed'],
    )

    # Define Prompt Template
    prompt_template = """Below is an inquiry from a banking customer. Classify the intent of the inquiry.

### Inquiry:
{}

### Intent:
{}"""

    def formatting_prompts_func(examples):
        inputs       = examples["text"]
        outputs      = examples["intent"]
        texts = []
        for input_text, output_text in zip(inputs, outputs):
            text = prompt_template.format(input_text, output_text)
            texts.append(text)
        return { "text" : texts, }

    # Load and Prepare Data
    df_train = pd.read_csv(config['data']['train_path'])
    dataset = Dataset.from_pandas(df_train)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # Trainer Setup
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = config['model']['max_seq_length'],
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = config['training']['batch_size'],
            gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
            warmup_steps = config['training']['warmup_steps'],
            max_steps = -1, # Set to -1 to use num_train_epochs
            num_train_epochs = config['training']['epochs'],
            learning_rate = config['training']['learning_rate'],
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = config['training']['optimizer'],
            weight_decay = config['training']['weight_decay'],
            lr_scheduler_type = config['training']['lr_scheduler_type'],
            seed = config['training']['seed'],
            output_dir = config['training']['output_dir'],
            report_to = "none",
            average_tokens_across_devices = False,  # Fix: Unsloth + Transformers >= 4.57
        ),
    )

    # Start Training
    print("Starting training...")
    trainer.train()

    # Save Model
    print(f"Saving model to {config['training']['output_dir']}...")
    model.save_pretrained(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])
    print("Training completed.")

if __name__ == "__main__":
    train()
