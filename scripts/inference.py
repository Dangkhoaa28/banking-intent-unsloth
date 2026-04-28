import os
import yaml
import torch


class IntentClassification:
    def __init__(self, model_path):
        """
        Load configuration file, tokenizer, and model checkpoint.
        
        Supports 3 modes (auto-detected):
        1. Unsloth + CUDA GPU (fastest)
        2. Transformers + PEFT + CUDA GPU (standard)
        3. Transformers + PEFT + CPU (slow but works anywhere)
        
        Args:
            model_path (str): Path to the configuration YAML file containing
                              at least the path to the saved model checkpoint.
        """
        # Load config
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        checkpoint_path = self.config['model']['checkpoint_path']
        max_seq_length = self.config['model'].get('max_seq_length', 1024)
        load_in_4bit = self.config['model'].get('load_in_4bit', True)
        
        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        # Try loading with Unsloth first (fastest on GPU)
        if self.device == "cuda":
            try:
                from unsloth import FastLanguageModel
                print(f"Loading model with Unsloth from {checkpoint_path}...")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=checkpoint_path,
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                )
                FastLanguageModel.for_inference(self.model)
                self._mode = "unsloth"
                print("Model loaded with Unsloth (GPU accelerated)")
                
            except (ImportError, Exception) as e:
                print(f"Unsloth not available ({e}), falling back to transformers+peft...")
                self._load_with_transformers(checkpoint_path, load_in_4bit)
        else:
            # CPU mode - use transformers + peft
            self._load_with_transformers(checkpoint_path, load_in_4bit=False)
        
        # Prompt template (must match training format exactly)
        self.prompt_template = """Below is an inquiry from a banking customer. Classify the intent of the inquiry.

### Inquiry:
{}

### Intent:
"""

    def _load_with_transformers(self, checkpoint_path, load_in_4bit=False):
        """Fallback: load model using standard transformers + peft."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        
        # Read base model name from adapter_config.json
        import json
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "unsloth/llama-3-8b-bnb-4bit")
        
        print(f"Loading base model: {base_model_name}")
        print(f"Loading LoRA adapter from: {checkpoint_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        # Load base model
        if load_in_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            # CPU mode: load in float32 (no quantization)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        self.model.eval()
        self._mode = f"transformers+peft ({self.device})"
        print(f"Model loaded with transformers+peft on {self.device}")
        if self.device == "cpu":
            print("CPU mode: inference will be slow (~30-60s per prediction)")

    def __call__(self, message):
        """
        Receive an input message and return the predicted intent label.
        
        Args:
            message (str): A banking customer inquiry text.
            
        Returns:
            str: The predicted intent label.
        """
        inputs = self.tokenizer(
            [self.prompt_template.format(message)],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=64,
                use_cache=True,
            )
        
        decoded_output = self.tokenizer.batch_decode(outputs)[0]
        
        # Extract the intent from the generated text
        try:
            predicted_label = decoded_output.split("### Intent:")[1]
            # Handle different end tokens
            for stop_token in [self.tokenizer.eos_token, "<|end_of_text|>", "<|eot_id|>", "\n\n"]:
                if stop_token and stop_token in predicted_label:
                    predicted_label = predicted_label.split(stop_token)[0]
            predicted_label = predicted_label.strip().split("\n")[0].strip()
        except (IndexError, AttributeError):
            predicted_label = "unknown"
            
        return predicted_label


# ============================================================
# Usage Example
# ============================================================
if __name__ == "__main__":
    import sys
    
    # Default config path
    config_path = "configs/inference.yaml"
    
    # Get message from command line or use default
    if len(sys.argv) > 1:
        msg = sys.argv[1]
    else:
        msg = "I want to open a new bank account."
    
    # Initialize the classifier
    print("=" * 60)
    print("BANKING INTENT CLASSIFICATION")
    print("=" * 60)
    
    classifier = IntentClassification(model_path=config_path)
    
    # Predict
    predicted_label = classifier(msg)
    
    print(f"\n{'=' * 60}")
    print(f"  Message:          {msg}")
    print(f"  Predicted Intent: {predicted_label}")
    print(f"  Mode:             {classifier._mode}")
    print(f"{'=' * 60}")
