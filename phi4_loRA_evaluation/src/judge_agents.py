import asyncio
import time
import logging
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.models import EvaluationState, JudgeResult, ModelConfig, PromptTemplates, create_default_judgment
from src.utils import safe_json_parse, handle_exception

logger = logging.getLogger(__name__)

class LightweightJudgeAgent:
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or ModelConfig.OLLAMA_BASE_URL
        self.models = ModelConfig.JUDGE_MODELS
        self.timeout = ModelConfig.TIMEOUT_SECONDS
        
        logger.info(f"Judge agent initialized with models: {list(self.models.keys())}")
    
    def _create_chat_model(self, model_name: str) -> ChatOllama:
        return ChatOllama(
            model=model_name,
            base_url=self.base_url,
            temperature=0.0,
            num_predict=4096,
            timeout=self.timeout
        )
    
    @retry(
        stop=stop_after_attempt(ModelConfig.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    @handle_exception
    async def _evaluate_single_judge(
        self, 
        model_key: str, 
        state: EvaluationState
    ) -> Dict[str, Any]:
        
        start_time = time.time()
        model_name = self.models[model_key]
        
        try:
            logger.info(f"Starting {model_key} ({model_name}) evaluation for task {state['task_id']}")
            
            chat_model = self._create_chat_model(model_name)
            
            prompt = PromptTemplates.JUDGE_PROMPT.format(
                input=state["input_text"],
                reference=state["reference_output"],
                base_output=state["base_output"],
                lora_output=state["lora_output"],
                base_tokens=state["base_tokens"],
                lora_tokens=state["lora_tokens"]
            )
            
            response = await chat_model.ainvoke([HumanMessage(content=prompt)])
            processing_time = time.time() - start_time
            
            result = safe_json_parse(
                response.content,
                create_default_judgment(model_key, "JSON解析エラー")
            )
            
            result["model_name"] = model_key
            result["processing_time"] = processing_time
            result["error"] = None
            
            logger.info(
                f"{model_key} evaluation completed in {processing_time:.2f}s "
                f"(task {state['task_id']}) - Choice: {result.get('better', 'unknown')}"
            )
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"{model_key} evaluation failed: {str(e)}"
            logger.error(f"{error_msg} after {processing_time:.2f}s")
            
            result = create_default_judgment(model_key, str(e))
            result["processing_time"] = processing_time
            return result
    
    async def evaluate_all_judges_parallel(self, state: EvaluationState) -> List[Dict[str, Any]]:
        logger.info(f"Starting parallel judge evaluation for task {state['task_id']}")
        start_time = time.time()
        
        judge_tasks = [
            self._evaluate_single_judge("deepseek", state),
            self._evaluate_single_judge("qwen", state),
            self._evaluate_single_judge("gemma", state)
        ]
        
        judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        processed_results = []
        for i, result in enumerate(judge_results):
            judge_key = list(self.models.keys())[i]
            
            if isinstance(result, Exception):
                logger.error(f"Judge {judge_key} failed with exception: {result}")
                error_result = create_default_judgment(judge_key, str(result))
                error_result["processing_time"] = 0.0
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        logger.info(
            f"All judges completed for task {state['task_id']} in {total_time:.2f}s"
        )
        return processed_results

judge_agent = LightweightJudgeAgent()

async def input_node(state: EvaluationState) -> EvaluationState:
    """Input node - validate and prepare state for evaluation."""
    logger.info(f"Processing input for task {state['task_id']}")

    required_fields = ["task_id", "input_text", "base_output", "lora_output"]
    for field in required_fields:
        if field == "task_id":
            if state.get(field) is None:
                error_msg = f"Missing required field: {field}"
                state["errors"].append(error_msg)
                logger.error(error_msg)
        else:
            if not state.get(field):
                error_msg = f"Missing required field: {field}"
                state["errors"].append(error_msg)
                logger.error(error_msg)
    
    if state["errors"]:
        state["status"] = "error"
        return state
    
    state["status"] = "ready_for_judges"
    logger.info(f"Input validation completed for task {state['task_id']}")
    
    return state

async def execute_all_judges(state: EvaluationState) -> EvaluationState:
    start_time = time.time()
    
    if state["status"] != "ready_for_judges":
        error_msg = f"Invalid state for judge execution: {state['status']}"
        state["errors"].append(error_msg)
        state["status"] = "error"
        logger.error(error_msg)
        return state
    try:
        state["status"] = "judges_processing"
        judge_results = await judge_agent.evaluate_all_judges_parallel(state)
        state["judge_results"] = judge_results
        successful_judges = sum(1 for result in judge_results if result and result.get("error") is None)
        failed_judges = len(judge_results) - successful_judges
        
        if successful_judges == 0:
            state["status"] = "judges_failed"
            state["errors"].append("All judges failed")
            logger.error(f"All judges failed for task {state['task_id']}")
        elif failed_judges > 0:
            state["status"] = "judges_partial"
            state["errors"].append(f"{failed_judges} judges failed")
            logger.warning(f"{failed_judges} judges failed for task {state['task_id']}")
        else:
            state["status"] = "judges_completed"
            logger.info(f"All judges succeeded for task {state['task_id']}")
        
        processing_time = time.time() - start_time
        state["processing_time"] += processing_time
        
        logger.info(
            f"Judge evaluation completed for task {state['task_id']} - "
            f"Success: {successful_judges}/{len(judge_results)} in {processing_time:.2f}s"
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Judge execution failed: {str(e)}"
        state["errors"].append(error_msg)
        state["status"] = "judges_failed"
        state["processing_time"] += processing_time
        logger.error(f"{error_msg} for task {state['task_id']}")
    
    return state

def get_judge_consensus(judge_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not judge_results:
        return {"consensus": "none", "confidence": 0.0, "pattern": "no_results"}
    
    votes = {"A": 0, "B": 0, "uncertain": 0}
    for result in judge_results:
        choice = result.get("better", "uncertain")
        if choice in votes:
            votes[choice] += 1
        else:
            votes["uncertain"] += 1
    
    total_votes = sum(votes.values())
    if total_votes == 0:
        return {"consensus": "none", "confidence": 0.0, "pattern": "invalid_votes"}
    
    max_votes = max(votes.values())
    winners = [choice for choice, count in votes.items() if count == max_votes]
    
    if len(winners) == 1 and max_votes > total_votes // 2:
        consensus = winners[0]
        confidence = max_votes / total_votes
        pattern = "majority" if confidence >= 0.67 else "plurality"
    elif len(winners) == 1:
        consensus = winners[0]
        confidence = max_votes / total_votes
        pattern = "weak_consensus"
    else:
        consensus = "tie"
        confidence = max_votes / total_votes
        pattern = "tie" 
    return {
        "consensus": consensus,
        "confidence": confidence,
        "pattern": pattern,
        "vote_breakdown": votes,
        "successful_judges": len([r for r in judge_results if r and r.get("error") is None])
    }