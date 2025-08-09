import time
import logging
from typing import Dict, Any, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.models import EvaluationState, ManagerResult, ModelConfig, PromptTemplates
from src.utils import safe_json_parse, handle_exception
from src.judge_agents import get_judge_consensus

logger = logging.getLogger(__name__)

class ManagerAgent:
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or ModelConfig.OLLAMA_BASE_URL
        self.model_name = ModelConfig.MANAGER_MODEL
        self.timeout = ModelConfig.TIMEOUT_SECONDS
        
        logger.info(f"Manager agent initialized with model: {self.model_name}")
    
    def _create_chat_model(self) -> ChatOllama:
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0,
            num_predict=2048,
            timeout=self.timeout
        )
    
    def _format_judge_results(self, judge_results: List[Dict[str, Any]]) -> str:
        formatted_results = []
        
        for result in judge_results:
            model_name = result.get("model_name", "unknown")
            better_choice = result.get("better", "uncertain")
            reason = result.get("reason", "理由なし")
            scores = result.get("scores", {})
            error = result.get("error")
            if error:
                formatted_results.append(
                    f"- {model_name}: エラー発生 ({error})"
                )
            else:
                formatted_results.append(
                    f"- {model_name}: 選択={better_choice}, "
                    f"スコア={scores}, 理由={reason}"
                )
        return "\n".join(formatted_results)
    
    def _create_default_manager_result(self, error_msg: str) -> Dict[str, Any]:
        return {
            "最終判断": "uncertain",
            "理由要約": f"管理者評価エラー: {error_msg}",
            "スコア内訳": {
                "簡潔さ": "Tie",
                "核心理解": "Tie", 
                "正確性": "Tie",
                "明快さ": "Tie"
            },
            "confidence": 0.0,
            "processing_time": 0.0,
            "error": error_msg
        }
    
    @retry(
        stop=stop_after_attempt(ModelConfig.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    @handle_exception
    async def integrate_judgments(
        self, 
        state: EvaluationState
    ) -> Dict[str, Any]:
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting manager integration for task {state['task_id']}")
            if not state.get("judge_results"):
                raise ValueError("No judge results available for integration")
            consensus_analysis = get_judge_consensus(state["judge_results"])
            logger.info(
                f"Judge consensus for task {state['task_id']}: "
                f"{consensus_analysis['consensus']} "
                f"(confidence: {consensus_analysis['confidence']:.2f})"
            )
            if (consensus_analysis["confidence"] >= 0.67 and 
                consensus_analysis["consensus"] in ["A", "B"]):
                processing_time = time.time() - start_time
                result = {
                    "最終判断": consensus_analysis["consensus"],
                    "理由要約": f"Judge間で強い合意 ({consensus_analysis['confidence']:.0%}の支持)",
                    "スコア内訳": self._derive_score_breakdown(state["judge_results"]),
                    "confidence": consensus_analysis["confidence"],
                    "processing_time": processing_time,
                    "error": None
                }
                logger.info(
                    f"Manager used consensus for task {state['task_id']}: "
                    f"{result['最終判断']} in {processing_time:.2f}s"
                )
                return result
            
            chat_model = self._create_chat_model()
            judge_results_text = self._format_judge_results(state["judge_results"])
            
            prompt = PromptTemplates.MANAGER_PROMPT.format(
                deepseek_result=self._get_judge_result_by_name("deepseek", state["judge_results"]),
                qwen3_result=self._get_judge_result_by_name("qwen", state["judge_results"]),
                gemma3_result=self._get_judge_result_by_name("gemma", state["judge_results"])
            )
            
            response = await chat_model.ainvoke([HumanMessage(content=prompt)])
            processing_time = time.time() - start_time
            
            result = safe_json_parse(
                response.content,
                self._create_default_manager_result("JSON解析エラー")
            )
            
            result["confidence"] = consensus_analysis["confidence"]
            result["processing_time"] = processing_time
            result["error"] = None
            
            logger.info(
                f"Manager integration completed for task {state['task_id']}: "
                f"{result.get('最終判断', 'unknown')} in {processing_time:.2f}s"
            )
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Manager integration failed: {str(e)}"
            logger.error(f"{error_msg} for task {state['task_id']}")
            result = self._create_default_manager_result(str(e))
            result["processing_time"] = processing_time
            return result
    
    def _get_judge_result_by_name(
        self, 
        model_name: str, 
        judge_results: List[Dict[str, Any]]
    ) -> str:

        for result in judge_results:
            if result.get("model_name") == model_name:
                if result.get("error"):
                    return f"エラー: {result['error']}"
                else:
                    return f"選択={result.get('better', 'uncertain')}, 理由={result.get('reason', '不明')}"
        return "結果なし"
    
    def _derive_score_breakdown(self, judge_results: List[Dict[str, Any]]) -> Dict[str, str]:
        categories = ["簡潔さ", "核心理解", "正確性", "明快さ"]
        score_mapping = {
            "conciseness": "簡潔さ",
            "core_understanding": "核心理解",
            "factual_accuracy": "正確性", 
            "clarity": "明快さ"
        }

        breakdown = {}
        
        for category in categories:
            eng_key = None
            for eng, jpn in score_mapping.items():
                if jpn == category:
                    eng_key = eng
                    break
            
            if not eng_key:
                breakdown[category] = "Tie"
                continue
            
            a_votes, b_votes = 0, 0
            
            for result in judge_results:
                if result.get("error"):
                    continue
                    
                scores = result.get("scores", {})
                choice = result.get("better", "uncertain")
                
                if choice == "A":
                    a_votes += 1
                elif choice == "B":
                    b_votes += 1
            
            if a_votes > b_votes:
                breakdown[category] = "A"
            elif b_votes > a_votes:
                breakdown[category] = "B"
            else:
                breakdown[category] = "Tie"
        
        return breakdown

manager_agent = ManagerAgent()

async def manager_node(state: EvaluationState) -> EvaluationState:
    start_time = time.time()
    
    if state["status"] not in ["judges_completed", "judges_partial"]:
        error_msg = f"Invalid state for manager: {state['status']}"
        state["errors"].append(error_msg)
        state["status"] = "manager_failed"
        logger.error(error_msg)
        return state
    
    try:
        state["status"] = "manager_processing"
        manager_result = await manager_agent.integrate_judgments(state)
        state["final_judgment"] = manager_result.get("最終判断")
        state["reason_summary"] = manager_result.get("理由要約")
        state["score_breakdown"] = manager_result.get("スコア内訳")
        
        if manager_result.get("error"):
            state["errors"].append(f"Manager error: {manager_result['error']}")
            state["status"] = "manager_failed"
        else:
            state["status"] = "completed"
        
        processing_time = time.time() - start_time
        state["processing_time"] += processing_time
        
        logger.info(
            f"Manager processing completed for task {state['task_id']} - "
            f"Decision: {state['final_judgment']} in {processing_time:.2f}s"
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Manager node failed: {str(e)}"
        state["errors"].append(error_msg)
        state["status"] = "manager_failed"
        state["processing_time"] += processing_time
        logger.error(f"{error_msg} for task {state['task_id']}")
    
    return state

async def output_node(state: EvaluationState) -> EvaluationState:
    logger.info(f"Finalizing output for task {state['task_id']}")
    
    if not state.get("final_judgment"):
        state["final_judgment"] = "uncertain"
        state["reason_summary"] = "評価プロセス未完了"
        state["errors"].append("No final judgment available")
    
    if state["status"] == "completed" and state["errors"]:
        state["status"] = "completed_with_warnings"
    elif state["status"] not in ["completed", "completed_with_warnings"]:
        state["status"] = "failed"
    
    logger.info(
        f"Task {state['task_id']} finalized - "
        f"Status: {state['status']}, Decision: {state['final_judgment']}, "
        f"Time: {state['processing_time']:.2f}s"
    )
    return state