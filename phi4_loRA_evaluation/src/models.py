from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

class EvaluationState(TypedDict):
    task_id: int
    input_text: str
    reference_output: str
    base_output: str
    lora_output: str
    base_tokens: int
    lora_tokens: int
    judge_results: List[Dict[str, Any]] 
    final_judgment: Optional[str] 
    reason_summary: Optional[str] 
    score_breakdown: Optional[Dict[str, str]] 
    processing_time: float  
    errors: List[str]  
    status: str 

@dataclass
class JudgeResult:
    model_name: str
    better_choice: str  # "A" or "B"
    scores: Dict[str, int]  # conciseness, core_understanding, factual_accuracy, clarity
    reason: str
    processing_time: float
    error: Optional[str] = None

@dataclass
class ManagerResult:
    final_judgment: str  # "A" or "B" or "uncertain"
    reason_summary: str
    score_breakdown: Dict[str, str]  # conciseness, core_understanding, factual_accuracy, clarity
    confidence: float
    processing_time: float
    error: Optional[str] = None

@dataclass
class BenchmarkTask:
    task_id: int
    input_text: str
    reference_output: str
    base_output: str
    lora_output: str
    base_generation_time: float
    lora_generation_time: float
    input_tokens: int
    base_output_tokens: int
    lora_output_tokens: int
    base_success: bool
    lora_success: bool
    timestamp: str

@dataclass
class EvaluationReport:
    metadata: Dict[str, Any]
    overall_statistics: Dict[str, Any]
    task_details: List[Dict[str, Any]]
    improvement_suggestions: Dict[str, List[str]]

class ModelConfig:
    JUDGE_MODELS = {
        "deepseek": "deepseek-r1:8b",    
        "qwen": "qwen3:8b",              
        "gemma": "gemma3:12b"             
    }
    MANAGER_MODEL = "gpt-oss:20b"        
    OLLAMA_BASE_URL = "http://ollama:11434"
    TIMEOUT_SECONDS = 120
    MAX_RETRIES = 3


class PromptTemplates:

    JUDGE_PROMPT = """
あなたは優秀なLLM評価者です。以下の2つの回答（AとB）を比較し、
**LoRAファインチューニングの目的**に照らして評価してください。

【ベースモデルをLoRAファインチューニングした真の目的】
ユーザーの問いに対して、「簡潔・明快に、質問意図を外さず、正確に回答できる」能力の向上

【評価観点】
1. 簡潔さと過不足のなさ（冗長でないか）
2. 質問の核心理解（問いの本質を捉えているか）
3. 正確性（事実・論理的誤りがないか）
4. 表現の整理度（明快な構成、読者に優しいか）

【質問】: {input}
【模範回答（参考）】: {reference}
【回答A（ベースモデル）】: {base_output} ({base_tokens}トークン)
【回答B（LoRAモデル）】: {lora_output} ({lora_tokens}トークン)

【出力形式（JSON）】:
{{
  "better": "A" または "B",
  "scores": {{
    "conciseness": [0-10],
    "core_understanding": [0-10], 
    "factual_accuracy": [0-10],
    "clarity": [0-10]
  }},
  "reason": "どちらがなぜ優れているかを100文字以内で説明"
}}
"""
    
    MANAGER_PROMPT = """
あなたは評価マネージャーAIです。以下のJudge評価に基づき、
LoRAファインチューニング目的に対して A か B がより目的に沿っているか判断してください。

【ベースモデルをLoRAファインチューニングした真の目的】
「モデルが、ユーザーの意図を過不足なく捉え、簡潔で明快かつ正確な回答を返す適性を高めたか」

【Judge評価結果】
- deepseek-r1: {deepseek_result}
- qwen3: {qwen3_result}
- gemma3: {gemma3_result}

【意見分岐時の判断基準】
議論や再評価は行わず、上記の「真の目的」に照らして統合判断を下してください。

【出力形式（JSON）】
{{
  "最終判断": "A" または "B" または "uncertain",
  "理由要約": "100文字以内の総合理由",
  "スコア内訳": {{
    "簡潔さ": "A" または "B" または "Tie",
    "核心理解": "A" または "B" または "Tie",
    "正確性": "A" または "B" または "Tie",
    "明快さ": "A" または "B" または "Tie"
  }}
}}
"""

def create_initial_state(task: BenchmarkTask) -> EvaluationState:
    return EvaluationState(
        task_id=task.task_id,
        input_text=task.input_text,
        reference_output=task.reference_output,
        base_output=task.base_output,
        lora_output=task.lora_output,
        base_tokens=task.base_output_tokens,
        lora_tokens=task.lora_output_tokens,
        judge_results=[],
        final_judgment=None,
        reason_summary=None,
        score_breakdown=None,
        processing_time=0.0,
        errors=[],
        status="processing"
    )

def create_default_judgment(model_name: str, error_msg: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "better": "uncertain",
        "scores": {
            "conciseness": 5,
            "core_understanding": 5,
            "factual_accuracy": 5,
            "clarity": 5
        },
        "reason": f"評価エラー: {error_msg}",
        "processing_time": 0.0,
        "error": error_msg
    }