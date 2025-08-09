import asyncio
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from src.models import EvaluationState, BenchmarkTask, create_initial_state
from src.judge_agents import input_node, execute_all_judges
from src.manager_agent import manager_node, output_node
from src.utils import create_progress_tracker

logger = logging.getLogger(__name__)

class EvaluationGraph:
    
    def __init__(self):
        self.graph = None
        self.compiled_graph = None
        self._build_graph()
        logger.info("Evaluation graph initialized")
    
    def _build_graph(self):
        logger.info("Building evaluation graph structure...")
        
        self.graph = StateGraph(EvaluationState)
        
        self.graph.add_node("input", input_node)
        self.graph.add_node("judges_parallel", execute_all_judges)
        self.graph.add_node("manager", manager_node)
        self.graph.add_node("output", output_node)
        
        self.graph.set_entry_point("input")
        self.graph.add_edge("input", "judges_parallel")
        self.graph.add_edge("judges_parallel", "manager") 
        self.graph.add_edge("manager", "output")
        self.graph.add_edge("output", END)
        
        self.compiled_graph = self.graph.compile()
        logger.info("Evaluation graph built and compiled successfully")
    
    async def evaluate_single_task(self, task: BenchmarkTask) -> EvaluationState:
        
        logger.info(f"Starting evaluation for task {task.task_id}")

        initial_state = create_initial_state(task)
        
        try:
            final_state = await self.compiled_graph.ainvoke(initial_state)
            status = final_state.get("status", "unknown")
            decision = final_state.get("final_judgment", "unknown")
            processing_time = final_state.get("processing_time", 0.0)
            
            logger.info(
                f"Task {task.task_id} completed - "
                f"Status: {status}, Decision: {decision}, "
                f"Time: {processing_time:.2f}s"
            )
            
            return final_state
            
        except Exception as e:
            error_msg = f"Graph execution failed for task {task.task_id}: {str(e)}"
            logger.error(error_msg)
            
            error_state = initial_state.copy()
            error_state["status"] = "graph_error"
            error_state["errors"].append(error_msg)
            error_state["final_judgment"] = "uncertain"
            error_state["reason_summary"] = f"グラフ実行エラー: {str(e)}"
            
            return error_state
    
    async def evaluate_batch_sequential(
        self, 
        tasks: List[BenchmarkTask],
        max_concurrent: int = 1
    ) -> List[EvaluationState]:
        
        logger.info(f"Starting batch evaluation of {len(tasks)} tasks (max_concurrent={max_concurrent})")
        
        results = []
        progress = create_progress_tracker(len(tasks), "Batch Evaluation")
    
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(task: BenchmarkTask) -> EvaluationState:
            async with semaphore:
                result = await self.evaluate_single_task(task)
                progress.update()
                return result

        if max_concurrent == 1:
            for task in tasks:
                result = await self.evaluate_single_task(task)
                results.append(result)
                progress.update()
        else:
            tasks_coroutines = [evaluate_with_semaphore(task) for task in tasks]
            results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {tasks[i].task_id} failed with exception: {result}")

                    error_state = create_initial_state(tasks[i])
                    error_state["status"] = "batch_error"
                    error_state["errors"].append(str(result))
                    error_state["final_judgment"] = "uncertain"
                    processed_results.append(error_state)
                else:
                    processed_results.append(result)
            
            results = processed_results
        
        progress.finish()

        successful = sum(1 for r in results if r and r.get("status") in ["completed", "completed_with_warnings"])
        failed = len(results) - successful
        total_time = sum(r.get("processing_time", 0.0) for r in results if r)
        avg_time = total_time / len(results) if results else 0.0
        
        logger.info(
            f"Batch evaluation completed: {successful}/{len(tasks)} successful, "
            f"{failed} failed, total time: {total_time:.1f}s, avg: {avg_time:.1f}s/task"
        )
        
        return results
    
    def get_evaluation_statistics(self, results: List[EvaluationState]) -> Dict[str, Any]:

        if not results:
            return {"error": "No results provided"}

        total_tasks = len(results)
        successful = sum(1 for r in results if r and r.get("status") in ["completed", "completed_with_warnings"])
        failed = total_tasks - successful
        
        lora_wins = sum(1 for r in results if r and r.get("final_judgment") == "B")
        base_wins = sum(1 for r in results if r and r.get("final_judgment") == "A")
        uncertain = sum(1 for r in results if r and r.get("final_judgment") in ["uncertain", None])
        
        processing_times = [r.get("processing_time", 0.0) for r in results if r]
        total_time = sum(processing_times)
        avg_time = total_time / total_tasks if total_tasks > 0 else 0.0
        
        score_categories = ["簡潔さ", "核心理解", "正確性", "明快さ"]
        category_stats = {}
        
        for category in score_categories:
            category_lora = 0
            category_base = 0
            category_tie = 0
            
            for result in results:
                if result:  
                    breakdown = result.get("score_breakdown", {})
                    choice = breakdown.get(category, "Tie")
                else:
                    choice = "Tie"
                
                if choice == "B":
                    category_lora += 1
                elif choice == "A":
                    category_base += 1
                else:
                    category_tie += 1
            
            category_stats[category] = {
                "LoRA勝利": category_lora,
                "ベース勝利": category_base,
                "同等": category_tie
            }
        
        error_types = {}
        for result in results:
            if result:
                for error in result.get("errors", []):
                    error_type = error.split(":")[0] if ":" in error else "その他"
                    error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "処理統計": {
                "総タスク数": total_tasks,
                "成功": successful,
                "失敗": failed,
                "成功率": f"{(successful / total_tasks * 100):.1f}%" if total_tasks > 0 else "0%"
            },
            "判定統計": {
                "LoRA勝利": lora_wins,
                "ベース勝利": base_wins,
                "判定保留": uncertain,
                "LoRA勝利率": f"{(lora_wins / total_tasks * 100):.1f}%" if total_tasks > 0 else "0%"
            },
            "パフォーマンス": {
                "総処理時間": f"{total_time:.1f}秒",
                "平均処理時間": f"{avg_time:.1f}秒/タスク",
                "処理速度": f"{total_tasks / total_time:.2f}タスク/秒" if total_time > 0 else "N/A"
            },
            "評価観点別統計": category_stats,
            "エラー分析": error_types
        }

evaluation_graph = EvaluationGraph()

async def run_single_evaluation(task: BenchmarkTask) -> EvaluationState:
    return await evaluation_graph.evaluate_single_task(task)

async def run_batch_evaluation(
    tasks: List[BenchmarkTask], 
    max_concurrent: int = 1
) -> List[EvaluationState]:
    return await evaluation_graph.evaluate_batch_sequential(tasks, max_concurrent)

def get_graph_statistics(results: List[EvaluationState]) -> Dict[str, Any]:
    return evaluation_graph.get_evaluation_statistics(results)