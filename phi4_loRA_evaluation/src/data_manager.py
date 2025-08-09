import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import glob

from src.models import BenchmarkTask, EvaluationState, EvaluationReport
from src.utils import (
    load_json_safely, 
    save_json_safely, 
    validate_benchmark_data, 
    format_timestamp,
    ensure_directory
)

logger = logging.getLogger(__name__)

class DataManager:
    
    def __init__(self, input_dir: str = "input", output_dir: str = "output", data_dir: str = "data"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        
        ensure_directory(str(self.input_dir))
        ensure_directory(str(self.output_dir))
        ensure_directory(str(self.data_dir))
        
        logger.info(f"DataManager initialized - Input: {self.input_dir}, Output: {self.output_dir}, Data: {self.data_dir}")
    
    def find_benchmark_files(self, pattern: str = "result_benchmark_*.json") -> List[Path]:
        search_pattern = str(self.input_dir / pattern)
        files = glob.glob(search_pattern)
        file_paths = [Path(f) for f in files]
        
        logger.info(f"Found {len(file_paths)} benchmark files matching '{pattern}'")
        return sorted(file_paths, key=lambda x: x.stat().st_mtime, reverse=True)  # Latest first
    
    def load_benchmark_data(self, file_path: Path) -> List[BenchmarkTask]:
        logger.info(f"Loading benchmark data from: {file_path}")
        
        raw_data = load_json_safely(str(file_path))
        if raw_data is None:
            raise ValueError(f"Failed to load benchmark data from {file_path}")
        
        validation_errors = validate_benchmark_data(raw_data)
        if validation_errors:
            error_msg = f"Benchmark data validation failed: {', '.join(validation_errors[:5])}"
            if len(validation_errors) > 5:
                error_msg += f" (and {len(validation_errors) - 5} more errors)"
            raise ValueError(error_msg)
        
        tasks = []
        for task_data in raw_data:
            try:
                task = BenchmarkTask(
                    task_id=task_data["task_id"],
                    input_text=task_data["input"],
                    reference_output=task_data.get("reference_output", ""),
                    base_output=task_data["base_output"],
                    lora_output=task_data["lora_output"],
                    base_generation_time=task_data.get("base_generation_time", 0.0),
                    lora_generation_time=task_data.get("lora_generation_time", 0.0),
                    input_tokens=task_data.get("input_tokens", 0),
                    base_output_tokens=task_data["base_output_tokens"],
                    lora_output_tokens=task_data["lora_output_tokens"],
                    base_success=task_data["base_success"],
                    lora_success=task_data["lora_success"],
                    timestamp=task_data.get("timestamp", "")
                )
                tasks.append(task)
            except Exception as e:
                logger.warning(f"Skipping invalid task {task_data.get('task_id', 'unknown')}: {e}")
        
        logger.info(f"Successfully loaded {len(tasks)} benchmark tasks")
        return tasks
    
    def save_intermediate_results(
        self, 
        results: List[EvaluationState], 
        filename: str = None
    ) -> Path:

        if filename is None:
            filename = f"intermediate_results_{format_timestamp()}.json"
        
        file_path = self.data_dir / filename
        
        serializable_results = []
        for result in results:
            if result:
                serializable_result = dict(result)
                serializable_results.append(serializable_result)
        
        success = save_json_safely(serializable_results, str(file_path))
        if success:
            logger.info(f"Intermediate results saved to: {file_path}")
            return file_path
        else:
            raise RuntimeError(f"Failed to save intermediate results to {file_path}")
    
    def generate_assessment_report(
        self, 
        results: List[EvaluationState], 
        input_file_path: Path,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        
        logger.info(f"Generating assessment report for {len(results)} results")
        
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r and r.get("status") in ["completed", "completed_with_warnings"])
        
        lora_wins = sum(1 for r in results if r and r.get("final_judgment") == "B")
        base_wins = sum(1 for r in results if r and r.get("final_judgment") == "A")
        uncertain = sum(1 for r in results if r and r.get("final_judgment") in ["uncertain", None])
        
        total_processing_time = sum(r.get("processing_time", 0.0) for r in results if r)
        avg_processing_time = total_processing_time / total_tasks if total_tasks > 0 else 0.0
        
        category_stats = self._analyze_score_breakdown(results)
        
        task_details = []
        for result in results:
            if result: 
                input_text = result.get("input_text", "")
                task_detail = {
                    "タスクID": result.get("task_id"),
                    "質問": input_text[:100] + ("..." if len(input_text) > 100 else ""),
                    "最終判断": result.get("final_judgment", "unknown"),
                    "理由要約": result.get("reason_summary", ""),
                    "スコア内訳": result.get("score_breakdown", {}),
                    "Judge詳細評価": self._format_judge_details(result.get("judge_results", [])),
                    "処理時間": f"{result.get('processing_time', 0.0):.2f}秒",
                    "状態": result.get("status", "unknown"),
                    "エラー": result.get("errors", [])
                }
                task_details.append(task_detail)

        improvement_suggestions = self._generate_improvement_suggestions(results)

        report = {
            "評価メタデータ": {
                "評価実行日時": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "評価システム": "LangGraph評価エージェントチーム v1.0",
                "入力ファイル": input_file_path.name,
                "処理タスク数": total_tasks,
                "評価モデル": {
                    "Judge_1": "deepseek-r1:8b",
                    "Judge_2": "qwen3:8b",
                    "Judge_3": "gemma3:12b",
                    "Manager": "gpt-oss:20b"
                },
                "処理時間": f"{total_processing_time:.1f}秒",
                "成功率": f"{(successful_tasks / total_tasks * 100):.0f}%" if total_tasks > 0 else "0%"
            },
            
            "全体統計": {
                "LoRA勝利": lora_wins,
                "ベース勝利": base_wins,
                "判定保留": uncertain,
                "勝利率": f"{(lora_wins / total_tasks * 100):.0f}%" if total_tasks > 0 else "0%",
                "平均処理時間": f"{avg_processing_time:.1f}秒/タスク",
                "評価観点別統計": category_stats
            },
            
            "タスク別詳細": task_details,
            
            "改善提案": improvement_suggestions
        }
        
        if metadata:
            report["評価メタデータ"].update(metadata)
        
        logger.info(f"Assessment report generated - LoRA wins: {lora_wins}/{total_tasks} ({lora_wins/total_tasks*100:.1f}%)")
        return report
    
    def save_assessment_report(
        self, 
        report: Dict[str, Any], 
        filename: str = None
    ) -> Path:
        
        if filename is None:
            filename = f"assessment_report_{format_timestamp()}.json"
        
        file_path = self.output_dir / filename
        
        success = save_json_safely(report, str(file_path))
        if success:
            logger.info(f"Assessment report saved to: {file_path}")
            return file_path
        else:
            raise RuntimeError(f"Failed to save assessment report to {file_path}")
    
    def _analyze_score_breakdown(self, results: List[EvaluationState]) -> Dict[str, Dict[str, int]]:

        categories = ["簡潔さ", "核心理解", "正確性", "明快さ"]
        category_stats = {}
        
        for category in categories:
            lora_wins = 0
            base_wins = 0
            ties = 0
            
            for result in results:
                if result:
                    breakdown = result.get("score_breakdown", {})
                    choice = breakdown.get(category, "Tie")
                else:
                    choice = "Tie"
                
                if choice == "B":
                    lora_wins += 1
                elif choice == "A":
                    base_wins += 1
                else:
                    ties += 1
            
            category_stats[category] = {
                "LoRA勝利": lora_wins,
                "ベース勝利": base_wins,
                "同等": ties
            }
        
        return category_stats
    
    def _format_judge_details(self, judge_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:

        formatted_details = {}
        
        model_mapping = {
            "deepseek": "deepseek-r1",
            "qwen": "qwen3",
            "gemma": "gemma3"
        }
        
        for result in judge_results:
            if result:
                model_name = result.get("model_name", "unknown")
                display_name = model_mapping.get(model_name, model_name)
                
                if result.get("error"):
                    formatted_details[display_name] = {
                        "エラー": result["error"]
                    }
                else:
                    formatted_details[display_name] = {
                        "優れた回答": result.get("better", "unknown"),
                        "スコア": result.get("scores", {}),
                        "理由": result.get("reason", "")
                    }
        
        return formatted_details
    
    def _generate_improvement_suggestions(self, results: List[EvaluationState]) -> Dict[str, List[str]]:

        total_tasks = len(results)
        lora_wins = sum(1 for r in results if r and r.get("final_judgment") == "B")
        
        suggestions = {
            "LoRAの強み": [],
            "改善余地": []
        }
        
        if lora_wins > total_tasks * 0.6:
            suggestions["LoRAの強み"].append("全体的に高い評価を獲得")
        
        category_analysis = self._analyze_score_breakdown(results)
        
        for category, stats in category_analysis.items():
            if stats["LoRA勝利"] > stats["ベース勝利"]:
                suggestions["LoRAの強み"].append(f"{category}で優れた性能")
            elif stats["ベース勝利"] > stats["LoRA勝利"]:
                suggestions["改善余地"].append(f"{category}の向上が必要")
        
        avg_base_tokens = 0
        avg_lora_tokens = 0
        valid_tasks = 0
        
        for result in results:
            if result:
                base_tokens = result.get("base_tokens", 0)
                lora_tokens = result.get("lora_tokens", 0)
            else:
                base_tokens = 0
                lora_tokens = 0
            
            if base_tokens > 0 and lora_tokens > 0:
                avg_base_tokens += base_tokens
                avg_lora_tokens += lora_tokens
                valid_tasks += 1
        
        if valid_tasks > 0:
            avg_base_tokens /= valid_tasks
            avg_lora_tokens /= valid_tasks
            
            if avg_lora_tokens < avg_base_tokens * 0.8:
                efficiency = (avg_base_tokens - avg_lora_tokens) / avg_base_tokens * 100
                suggestions["LoRAの強み"].append(f"冗長性の大幅削減（平均{efficiency:.0f}%のトークン効率改善）")

        if not suggestions["LoRAの強み"]:
            suggestions["LoRAの強み"] = ["詳細分析のため更多データが必要"]
        
        if not suggestions["改善余地"]:
            suggestions["改善余地"] = ["現在の性能を維持"]
        
        return suggestions

data_manager = DataManager()

def load_benchmark_from_pattern(pattern: str = "result_benchmark_*.json") -> List[BenchmarkTask]:
    
    files = data_manager.find_benchmark_files(pattern)
    if not files:
        raise FileNotFoundError(f"No benchmark files found matching pattern: {pattern}")
    
    return data_manager.load_benchmark_data(files[0]) 

def save_evaluation_report(
    results: List[EvaluationState], 
    input_file_path: Path,
    metadata: Dict[str, Any] = None
) -> Path:

    report = data_manager.generate_assessment_report(results, input_file_path, metadata)
    return data_manager.save_assessment_report(report)