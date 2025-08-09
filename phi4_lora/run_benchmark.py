import os
import json
import torch
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import load_dataset

BASE_MODEL_NAME = "unsloth/Phi-4-unsloth-bnb-4bit"
LORA_MODEL_DIR = "lora_model"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.9
DO_SAMPLE = False
REPETITION_PENALTY = 1.1

class Benchmark:
    def __init__(self):
        self.results = []
        self.base_model = None
        self.base_tokenizer = None
        self.lora_model = None
        self.lora_tokenizer = None
    
    def load_base_model(self):
        print("Loading base model...")
        self.base_model, self.base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.base_model)
        print("Base model loaded")
    
    def load_lora_model(self):
        print("Loading LoRA model...")
        try:
            self.lora_model, self.lora_tokenizer = FastLanguageModel.from_pretrained(
                model_name=LORA_MODEL_DIR,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=LOAD_IN_4BIT,
            )
            FastLanguageModel.for_inference(self.lora_model)
            print("LoRA model loaded")
            return True
        except Exception as e:
            print(f"Error loading LoRA model: {e}")
            return False
    
    def load_elyza_dataset(self):
        print("Loading ELYZA-tasks-100 dataset...")
        try:
            dataset = load_dataset("elyza/ELYZA-tasks-100", split="test")
            print(f"Dataset loaded: {len(dataset)} tasks")
            print("Dataset features:", dataset.features)
            if len(dataset) > 0:
                print("First example keys:", list(dataset[0].keys()))
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def generate_response(self, model, tokenizer, prompt: str, model_name: str) -> Dict[str, Any]:
        try:
            if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
                if tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                    tokenizer.pad_token_id = tokenizer.unk_token_id
                else:
                    tokenizer.pad_token = "<pad>"
                    tokenizer.pad_token_id = getattr(tokenizer, 'vocab_size', 0)
            
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            attention_mask = torch.ones_like(inputs)
            inputs = inputs.to("cuda")
            attention_mask = attention_mask.to("cuda")

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=DO_SAMPLE,
                    repetition_penalty=REPETITION_PENALTY,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            end_time = time.time()
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return {
                "model_name": model_name,
                "response": response.strip(),
                "generation_time": end_time - start_time,
                "input_tokens": inputs.shape[1],
                "output_tokens": outputs.shape[1] - inputs.shape[1],
                "success": True
            }
        except Exception as e:
            print(f"Generation error for {model_name}: {e}")
            return {
                "model_name": model_name,
                "response": f"ERROR: {str(e)}",
                "generation_time": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "success": False
            }
    
    def run_benchmark(self, dataset, max_tasks: int = None):
        print("=" * 80)
        print("SIMPLE BENCHMARK - ELYZA-tasks-100")
        print("=" * 80)
        
        if not dataset:
            print("Dataset not available")
            return None
        
        self.load_base_model()
        if not self.load_lora_model():
            print("Skipping LoRA model evaluation")
            return None
        
        tasks_to_eval = min(len(dataset), max_tasks) if max_tasks else len(dataset)
        print(f"Processing {tasks_to_eval} tasks...")
        
        for i in range(tasks_to_eval):
            task = dataset[i]
            try:
                print(f"\n--- Task {i + 1}/{tasks_to_eval} ---")
                print(f"Input: {task['input'][:100]}..." if len(task['input']) > 100 else f"Input: {task['input']}")

                base_result = self.generate_response(
                    self.base_model, self.base_tokenizer, task['input'], "base_model"
                )
                lora_result = self.generate_response(
                    self.lora_model, self.lora_tokenizer, task['input'], "lora_model"
                )
                result = {
                    "task_id": i,
                    "input": task['input'],
                    "reference_output": task.get('reference_output', ''),
                    "base_output": base_result['response'] if base_result['success'] else None,
                    "lora_output": lora_result['response'] if lora_result['success'] else None,
                    "base_generation_time": base_result['generation_time'],
                    "lora_generation_time": lora_result['generation_time'],
                    "input_tokens": base_result['input_tokens'], 
                    "base_output_tokens": base_result['output_tokens'],
                    "lora_output_tokens": lora_result['output_tokens'],
                    "base_success": base_result['success'],
                    "lora_success": lora_result['success'],
                    "timestamp": datetime.now().isoformat()
                }
                self.results.append(result)  
                print(f"Base: {base_result['output_tokens']} tokens ({base_result['generation_time']:.2f}s)")
                print(f"LoRA: {lora_result['output_tokens']} tokens ({lora_result['generation_time']:.2f}s)")
            except Exception as e:
                print(f"Error in task {i + 1}: {e}")
                continue
    
    def save_results(self, filename: str = None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_benchmark_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {filename}")
        return filename

def main():
    try:
        benchmark = Benchmark()
        dataset = benchmark.load_elyza_dataset()
        if not dataset:
            return 
        benchmark.run_benchmark(dataset, max_tasks=100)  
        json_filename = benchmark.save_results()
        print(f"JSON Results: {json_filename}")  
    except Exception as e:
        print(f"JSON failed: {e}")
        raise

if __name__ == "__main__":
    main()