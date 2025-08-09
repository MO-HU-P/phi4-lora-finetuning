import json
import re
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import traceback
import psutil
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def safe_json_parse(text: str, fallback_result: Dict[str, Any]) -> Dict[str, Any]:
    if not text or not text.strip():
        logger.warning("Empty text provided for JSON parsing")
        return fallback_result
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'(\{[^{}]*\{[^{}]*\}[^{}]*\})',
        r'(\{.*?\})'
    ]
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match.strip())
                if isinstance(result, dict):
                    logger.info(f"Successfully extracted JSON using pattern: {pattern}")
                    return result
            except json.JSONDecodeError:
                continue
    try:
        cleaned_text = re.sub(r',\s*}', '}', text)
        cleaned_text = re.sub(r',\s*]', ']', cleaned_text)
        
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            logger.info("Successfully parsed JSON after cleanup")
            return result
    except (json.JSONDecodeError, AttributeError):
        pass
    
    logger.error(f"Failed to parse JSON from text: {text[:200]}...")
    return fallback_result

def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return None

def format_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_gpu_memory_usage() -> Dict[str, Any]:
    try:
        import py3nvml.py3nvml as nvml
        nvml.nvmlInit()

        handle = nvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        
        return {
            "total_gb": info.total / (1024**3),
            "used_gb": info.used / (1024**3),
            "free_gb": info.free / (1024**3),
            "usage_percent": (info.used / info.total) * 100
        }
    except Exception as e:
        logger.warning(f"Could not get GPU memory usage: {e}")
        return {"error": str(e)}


def get_system_resources() -> Dict[str, Any]:
    try:
        gpu_info = get_gpu_memory_usage()
        if "error" in gpu_info:
            gpu_info = {
                "total_gb": None,
                "used_gb": None,
                "free_gb": None,
                "usage_percent": None,
                "error": gpu_info["error"]
            }
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_info": gpu_info
        }
    except Exception as e:
        return {
            "cpu_percent": None,
            "memory_percent": None,
            "memory_used_gb": None,
            "memory_total_gb": None,
            "gpu_info": {"error": f"Failed to collect system resources: {e}"}
        }

async def measure_execution_time(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise e
    return wrapper

def handle_exception(func):
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            raise Exception(error_msg) from e
    
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            raise Exception(error_msg) from e
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def validate_benchmark_data(data: List[Dict[str, Any]]) -> List[str]:
    
    errors = []
    
    if not isinstance(data, list):
        errors.append("Benchmark data must be a list")
        return errors
    
    required_fields = [
        "task_id", "input", "base_output", "lora_output",
        "base_output_tokens", "lora_output_tokens", "base_success", "lora_success"
    ]
    
    for i, task in enumerate(data):
        if not isinstance(task, dict):
            errors.append(f"Task {i} is not a dictionary")
            continue
            
        for field in required_fields:
            if field not in task:
                errors.append(f"Task {i} missing required field: {field}")
            elif field.endswith("_tokens") and not isinstance(task.get(field), int):
                errors.append(f"Task {i} field {field} must be an integer")
            elif field.endswith("_success") and not isinstance(task.get(field), bool):
                errors.append(f"Task {i} field {field} must be a boolean")
    return errors

def create_progress_tracker(total_tasks: int, description: str = "Processing"):

    class ProgressTracker:
        def __init__(self, total: int, desc: str):
            self.total = total
            self.completed = 0
            self.description = desc
            self.start_time = time.time()
        
        def update(self, increment: int = 1):
            self.completed += increment
            elapsed_time = time.time() - self.start_time
            rate = self.completed / elapsed_time if elapsed_time > 0 else 0
            
            progress_percent = (self.completed / self.total) * 100
            eta = (self.total - self.completed) / rate if rate > 0 else 0
            
            logger.info(
                f"{self.description}: {self.completed}/{self.total} "
                f"({progress_percent:.1f}%) - {rate:.2f} tasks/sec - ETA: {eta:.1f}s"
            )
        
        def finish(self):
            total_time = time.time() - self.start_time
            rate = self.total / total_time if total_time > 0 else 0
            logger.info(
                f"{self.description} completed: {self.total} tasks in "
                f"{total_time:.1f}s (avg {rate:.2f} tasks/sec)"
            )
    return ProgressTracker(total_tasks, description)

def ensure_directory(path: str) -> Path:

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def save_json_safely(data: Dict[str, Any], file_path: str) -> bool:
    
    try:
        path_obj = Path(file_path)
        ensure_directory(str(path_obj.parent))
        
        if path_obj.exists():
            backup_path = path_obj.with_suffix(f".backup_{format_timestamp()}{path_obj.suffix}")
            path_obj.rename(backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        temp_path = path_obj.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            json.load(f)  
        
        temp_path.rename(path_obj)
        logger.info(f"Successfully saved JSON to: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False

def load_json_safely(file_path: str) -> Optional[Dict[str, Any]]:

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return None