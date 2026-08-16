import os
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
OUTPUT_DIR = "./lora_adapter"

def train():
    print(f"[Fine-Tuning] Loading base model '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Load pre-trained base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )

    # 2. Configure Parameter-Efficient Fine-Tuning (LoRA)
    lora_config = LoraConfig(
        r=8,                            # Rank dimension (controls matrix size)
        lora_alpha=16,                  # Scaling factor
        target_modules=["q_proj", "v_proj"], # Target attention modules
        lora_dropout=0.05,              # Dropout probability
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Wrap base model with trainable LoRA layers
    model = get_peft_model(model, lora_config)
    
    print("\n--- Trainable Parameter Summary ---")
    model.print_trainable_parameters()
    print("-----------------------------------\n")

    # 3. Create Sample Domain Instruction Dataset
    sample_data = [
        {
            "prompt": "User: What is key-value caching in LLMs?\nAssistant:",
            "response": " KV caching stores key and value matrices of previously processed tokens to avoid redundant recomputation during text generation."
        },
        {
            "prompt": "User: How does Low-Rank Adaptation (LoRA) reduce GPU memory?\nAssistant:",
            "response": " LoRA freezes pre-trained model weights and injects small trainable rank decomposition matrices, reducing trainable parameters by over 99%."
        },
        {
            "prompt": "User: Explain QLoRA.\nAssistant:",
            "response": " QLoRA quantizes the base model to 4-bit precision while keeping LoRA adapters at 16-bit to dramatically lower VRAM requirements during fine-tuning."
        }
    ]

    def process_data(examples):
        texts = [p + r for p, r in zip(examples["prompt"], examples["response"])]
        return tokenizer(texts, truncation=True, padding=True)

    dataset = Dataset.from_list(sample_data)
    tokenized_dataset = dataset.map(process_data, batched=True)

    # 4. Configure Training Environment
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        fp16=(device == "cuda"),
        use_cpu=(device == "cpu"),
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # 5. Train & Save LoRA Adapter
    print("[Fine-Tuning] Executing LoRA adaptation step...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[Fine-Tuning] Success! LoRA adapter weights saved to '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    train()