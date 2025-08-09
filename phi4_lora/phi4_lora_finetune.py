import os
import torch
import numpy as np
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq, TextStreamer, get_cosine_schedule_with_warmup


MODEL_NAME = "unsloth/Phi-4-unsloth-bnb-4bit"
DATASET_NAME = "msfm/ichikara-instruction-all"

OUTPUT_DIR = "outputs"
LORA_MODEL_DIR = "lora_model"

FINAL_MODEL_NAME = "phi-4-finetune"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

LORA_R = 8              
LORA_ALPHA = 16          
LORA_DROPOUT = 0.05         

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4    
WARMUP_STEPS = 10                  
NUM_TRAIN_EPOCHS = 1.3               
LEARNING_RATE = 5e-5               

WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0                

def load_model_and_tokenizer():
    print("Loading model and tokenizer...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,                               
        target_modules=["q_proj", "v_proj", "o_proj",],  
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,              
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=False,                        
        loftq_config=None,
    )
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    print("Loading and preparing dataset...")
    
    dataset = load_dataset(DATASET_NAME, split="train")
    
    print(f"Original dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print("Dataset keys:", list(dataset[0].keys()))
        print("First example:", dataset[0])

    tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
    
    def formatting_prompts_func(examples):
        texts = []
        for i in range(len(examples["text"])):
            conversation = [
                {"role": "user", "content": examples["text"][i]},
                {"role": "assistant", "content": examples["output"][i]}
            ]
            formatted_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            texts.append(formatted_text)
        
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    dataset_split = dataset.train_test_split(test_size=0.2, seed=3407)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    validate_dataset(train_dataset)
    
    return train_dataset, eval_dataset, tokenizer

def setup_trainer(model, tokenizer, train_dataset, eval_dataset):
    print("Setting up trainer...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,          
            logging_steps=10,                      
            eval_strategy="steps",                
            eval_steps=100,                       
            save_steps=100,                       
            optim="adamw_8bit",
            lr_scheduler_type="cosine_with_restarts",           
            seed=3407,
            output_dir=OUTPUT_DIR,
            dataloader_drop_last=True,            
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
    )
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user<|im_sep|>",
        response_part="<|im_start|>assistant<|im_sep|>",
    )

    print("Verifying response masking...")
    if len(trainer.train_dataset) > 5:
        input_tokens = tokenizer.decode(trainer.train_dataset[5]["input_ids"])
        space = tokenizer(" ", add_special_tokens=False).input_ids[0]
        masked_labels = tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
        print("Input tokens:", input_tokens[:200] + "...")
        print("Masked labels:", masked_labels[:200] + "...")
    
    return trainer

def train_model(trainer):
    print("Starting training...")
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer_stats = trainer.train()
    
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    print(f"\nTraining completed!")
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    return trainer_stats

def save_models(model, tokenizer):
    print("Saving models...")
    print("Saving LoRA adapters...")
    model.save_pretrained(LORA_MODEL_DIR)
    tokenizer.save_pretrained(LORA_MODEL_DIR)
    print(f"LoRA adapters saved to: {LORA_MODEL_DIR}")
    print("LoRA model saving completed - ready for inference!")

def validate_dataset(dataset):
    print("Validating dataset...")
    
    lengths = [len(item['text']) for item in dataset]
    print(f"Average length: {np.mean(lengths):.0f}")
    print(f"Maximum length: {max(lengths)}")
    print(f"Minimum length: {min(lengths)}")
    
    texts = [item['text'] for item in dataset]
    unique_texts = set(texts)
    print(f"Duplication rate: {(len(texts) - len(unique_texts)) / len(texts) * 100:.1f}%")
    
    return dataset

def load_saved_model():
    print("Loading saved LoRA model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def main():
    print("Starting Phi-4 LoRA Fine-tuning Pipeline")
    print("=" * 50)
    
    try:
        model, tokenizer = load_model_and_tokenizer()
        train_dataset, eval_dataset, tokenizer = prepare_dataset(tokenizer)
        trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset)
        trainer_stats = train_model(trainer)
        save_models(model, tokenizer)
        
        print("\n" + "=" * 50)
        print("Fine-tuning completed successfully!")
        print(f"LoRA adapters saved to: {LORA_MODEL_DIR}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()