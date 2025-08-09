import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional
import time
from src.data_manager import DataManager, load_benchmark_from_pattern, save_evaluation_report
from src.evaluation_graph import run_batch_evaluation, get_graph_statistics
from src.utils import get_system_resources, format_timestamp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"evaluation_{format_timestamp()}.log")
    ]
)
logger = logging.getLogger(__name__)

class EvaluationOrchestrator:
    
    def __init__(self, input_dir: str, output_dir: str, data_dir: str = "data"):
        self.data_manager = DataManager(input_dir, output_dir, data_dir)
        self.start_time = None
        
        logger.info(f"Evaluation orchestrator initialized")
        logger.info(f"Input: {input_dir}, Output: {output_dir}, Data: {data_dir}")
    
    async def run_evaluation(
        self,
        input_pattern: str = "result_benchmark_*.json",
        max_concurrent: int = 1,
        save_intermediate: bool = True
    ) -> bool:
        
        self.start_time = time.time()
        
        try:
            resources = get_system_resources()
            logger.info(f"System resources at start: {resources}")
            logger.info("Phase 1: Loading benchmark data...")
            benchmark_files = self.data_manager.find_benchmark_files(input_pattern)
            
            if not benchmark_files:
                logger.error(f"No benchmark files found matching pattern: {input_pattern}")
                return False
            
            input_file = benchmark_files[0] 
            logger.info(f"Using benchmark file: {input_file}")
            
            tasks = self.data_manager.load_benchmark_data(input_file)
            if not tasks:
                logger.error("No valid tasks found in benchmark file")
                return False
            
            logger.info(f"Loaded {len(tasks)} benchmark tasks")
            logger.info(f"Phase 2: Executing evaluation (max_concurrent={max_concurrent})...")

            if not await self._check_ollama_connection():
                logger.error("Cannot connect to Ollama service")
                return False

            results = await run_batch_evaluation(tasks, max_concurrent)
            
            if not results:
                logger.error("No evaluation results generated")
                return False
            
            logger.info(f"Evaluation completed: {len(results)} results")
            
            if save_intermediate:
                logger.info("Phase 3: Saving intermediate results...")
                intermediate_path = self.data_manager.save_intermediate_results(results)
                logger.info(f"Intermediate results saved: {intermediate_path}")
            
            logger.info("Phase 4: Generating assessment report...")
            
            statistics = get_graph_statistics(results)

            metadata = {
                "システム情報": get_system_resources(),
                "実行設定": {
                    "最大並行数": max_concurrent,
                    "入力パターン": input_pattern,
                    "中間保存": save_intermediate
                }
            }
            
            report_path = save_evaluation_report(results, input_file, metadata)
            
            total_time = time.time() - self.start_time
            self._log_final_summary(results, statistics, total_time, report_path)

            return True
            
        except Exception as e:
            total_time = time.time() - self.start_time if self.start_time else 0
            logger.error(f"Evaluation failed after {total_time:.1f}s: {str(e)}")
            return False
    
    async def _check_ollama_connection(self) -> bool:
        """Check if Ollama service is accessible."""
        try:
            from langchain_ollama import ChatOllama
            chat_model = ChatOllama(
                model="gemma3:12b",  # Test model
                base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                timeout=30
            )
            from langchain_core.messages import HumanMessage
            response = await chat_model.ainvoke([HumanMessage(content="Hello")])
            logger.info("Ollama connection verified")
            return True
        except Exception as e:
            logger.error(f"Ollama connection failed: {str(e)}")
            return False
    
    def _log_final_summary(
        self, 
        results, 
        statistics, 
        total_time: float, 
        report_path: Path
    ):
        valid_results = [r for r in results if r and isinstance(r, dict)]

        successful = sum(1 for r in valid_results if r.get("status") in ["completed", "completed_with_warnings"])
        lora_wins = sum(1 for r in valid_results if r.get("final_judgment") == "B")
        
        logger.info("============ EVALUATION COMPLETED ============")
        logger.info(f"Results Summary:")
        logger.info(f"   • Total Tasks: {len(results)}")
        logger.info(f"   • Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        logger.info(f"   • LoRA Wins: {lora_wins}/{len(results)} ({lora_wins/len(results)*100:.1f}%)")
        logger.info(f"   • Total Time: {total_time:.1f}s")
        logger.info(f"   • Avg Time/Task: {total_time/len(results):.1f}s")
        logger.info(f"Report saved to: {report_path}")
        logger.info("===============================================")

async def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Evaluation Agent Team - LoRA vs Base Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --input input/result_benchmark_*.json --output output/
  python src/main.py --max-concurrent 2 --no-intermediate
  python src/main.py --input-dir /app/input --output-dir /app/output
        """
    )
    
    parser.add_argument(
        "--input", 
        type=str,
        default="result_benchmark_*.json",
        help="Input file pattern or specific file path (default: result_benchmark_*.json)"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Input directory path (default: input)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory path (default: output)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory for intermediate files (default: data)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent evaluations (default: 1 for resource management)"
    )
    
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Skip saving intermediate results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce logging output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    if args.max_concurrent < 1:
        logger.error("max-concurrent must be at least 1")
        return 1
    
    if args.max_concurrent > 3:
        logger.warning("max-concurrent > 3 may cause GPU memory issues")
    
    try:
        orchestrator = EvaluationOrchestrator(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            data_dir=args.data_dir
        )
        success = await orchestrator.run_evaluation(
            input_pattern=args.input,
            max_concurrent=args.max_concurrent,
            save_intermediate=not args.no_intermediate
        )
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)