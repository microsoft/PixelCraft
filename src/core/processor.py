import os
import sys
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger
from tqdm import tqdm

from src.planners.action_planner import ActionPlanner
from src.utils.evaluate import compute_acc_single


# Setup logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    "logs/processor_{time:YYYY-MM-DD_HH-mm-ss}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="10 days",
    level="INFO",
)


def process_single_task(
    task_id: str,
    task_data: dict,
    output_dir: str,
    data_dir_root: str,
    task_type: str,
    max_steps: int = 40,
    overwrite: bool = False,
    tool_selection_json: dict = None,
) -> dict:
    """Process single task in a separate process"""
    try:
        # Re-configure logger for worker process
        logger.remove()
        logger.add(
            sys.stdout,
            format=f"<green>{{time:HH:mm:ss}}</green> | <cyan>Worker-{os.getpid()}</cyan> | <level>{{message}}</level>",
            level="INFO",
            colorize=True
        )

        # Create output directory
        task_output_dir = Path(output_dir) / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Skip if results exist and not overwriting
        answer_file = task_output_dir / "answer.json"
        if not overwrite and answer_file.exists():
            try:
                with open(answer_file) as f:
                    result = json.load(f)
                    return {
                        "task_id": task_id,
                        "accuracy": result["acc"],
                        "status": "skipped",
                    }
            except Exception:
                pass

        # Check tool selection skip conditions
        if tool_selection_json and task_id in tool_selection_json:
            adjustment = tool_selection_json[task_id].get("adjustment", "")
            if "True" not in adjustment and "initial_acc" in tool_selection_json[task_id]:
                acc = tool_selection_json[task_id]["initial_acc"]
                return {"task_id": task_id, "accuracy": acc, "status": "skipped"}

        # Prepare task data
        task_data = task_data.copy()
        task_data["figure_path"] = data_dir_root + task_data["figure_path"]

        # Get tool selection for this task
        tool_json = (
            tool_selection_json.get(task_id) if tool_selection_json else None
        )
        if tool_selection_json and len(tool_json["tools"]) <= 1:
            direct_answer = True
        else:
            direct_answer = False

        # Initialize components for this task
        action_planner = ActionPlanner(name="action_planner", task_type=task_type)

        try:
            # Run action planner
            response, all_messages, tool_history = action_planner.react(
                question=task_data["prompt"],
                image_path=task_data["figure_path"],
                output_dir=str(task_output_dir),
                max_steps=max_steps,
                tool_selection_json=tool_json,
                original_task=task_data["query"],
                direct_answer=direct_answer
            )

            # Compute accuracy
            accuracy, prediction, solution = compute_acc_single(task_data, response)

            # Save results
            # Save all messages
            with open(task_output_dir / "all_message.json", "w") as f:
                json.dump(all_messages, f, indent=2)

            # Save final answer
            with open(task_output_dir / "answer.json", "w") as f:
                json.dump(
                    {
                        "question": task_data["query"],
                        "answer": response,
                        "acc": accuracy,
                        "solution": solution,
                        "prediction": prediction,
                        "figure_path": task_data["figure_path"],
                        "tool_call_history": tool_history,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Task {task_id} completed with accuracy: {accuracy}")
            return {"task_id": task_id, "accuracy": accuracy, "status": "success"}

        finally:
            # Cleanup resources
            if action_planner:
                action_planner.reset()

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        return {
            "task_id": task_id,
            "accuracy": 0.0,
            "status": "error",
            "error": str(e),
        }


class TaskProcessor:
    """Multi-process task processor for handling chart/geo analysis tasks"""

    def __init__(self, task_type: str = "chart", data_dir_root: str = ""):
        """Initialize task processor with components"""
        self.task_type = task_type
        self.data_dir_root = data_dir_root

    def process_batch(
        self, tasks: dict, output_dir: str, max_concurrent: int = 5, **kwargs
    ) -> dict:
        """Process multiple tasks concurrently using processes"""
        results = []
        total = len(tasks)

        logger.info(f"Starting batch processing of {total} tasks with max_concurrent={max_concurrent}")

        # Create tqdm progress bar
        pbar = tqdm(
            total=total,
            desc="Processing tasks",
            unit="task",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Acc: {postfix}\n"
        )

        # Use ProcessPoolExecutor for concurrent processing
        with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all tasks
            futures = []
            for task_id, task_data in tasks.items():
                # if task_id != "532":
                #     continue
                future = executor.submit(
                    process_single_task,
                    task_id=task_id,
                    task_data=task_data,
                    output_dir=output_dir,
                    data_dir_root=self.data_dir_root,
                    task_type=self.task_type,
                    **kwargs
                )
                futures.append(future)
            
            # Wait for all to complete and update progress
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Calculate current accuracy
                    correct_count = sum(1 for r in results if r.get("accuracy", 0) == 1.0)
                    current_accuracy = correct_count / len(results) if results else 0.0
                    
                    # Update progress bar with accuracy
                    pbar.set_postfix_str(f"{current_accuracy:.2%}")
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Task execution failed with exception: {e}")
                    pbar.update(1)
        
        pbar.close()

        return self._compile_batch_results(results)

    def _compile_batch_results(self, results: list) -> dict:
        """Compile batch processing statistics"""
        successful = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errors = sum(1 for r in results if r["status"] == "error")

        accuracies = [r["accuracy"] for r in results]
        final_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

        return {
            "total_tasks": len(results),
            "successful_tasks": successful,
            "skipped_tasks": skipped,
            "failed_tasks": errors,
            "accuracy": final_accuracy,
            "all_accuracies": accuracies,
            "results": results,
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Multi-threaded chart/geo task processor")
    parser.add_argument("--data_path", type=str, default="data/chartxiv_val.json")
    parser.add_argument("--data_dir_root", type=str, default="./")
    parser.add_argument("--output_dir", type=str, default="eval")
    parser.add_argument("--output_dir_root", type=str, default="output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini_2025-04-14")
    parser.add_argument(
        "--task_type", type=str, default="chart", choices=["chart", "geo"]
    )
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--tool_selection_path", type=str, default="data/charxiv_selection.json")
    args = parser.parse_args()

    # Setup environment
    os.environ["DEPLOYMENT_NAME"] = args.model_name
    logger.info(f"Starting multi-threaded processing with model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load data
    with open(args.data_path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} tasks from {args.data_path}")

    # Load tool selection if provided
    if args.tool_selection_path and os.path.exists(args.tool_selection_path):
        with open(args.tool_selection_path) as f:
            tool_selection_json = json.load(f)
        logger.info(f"Loaded tool selection for {len(tool_selection_json)} tasks")
    else:
        tool_selection_json = None
        logger.warning("Tool selection path not provided or file does not exist.")

    # Setup output directory
    dataset_name = Path(args.data_path).stem
    output_path = (
        Path(args.output_dir_root)
        / f"{dataset_name}"
        / f"{args.model_name}-{args.output_dir}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Initialize processor
    processor = TaskProcessor(
        task_type=args.task_type, data_dir_root=args.data_dir_root
    )

    # Run batch processing
    results = processor.process_batch(
        tasks=data,
        output_dir=str(output_path),
        max_concurrent=args.max_concurrent,
        max_steps=args.max_steps,
        overwrite=args.overwrite,
        tool_selection_json=tool_selection_json,
    )

    # Save results
    save_final_results(output_path, results, args)

    # Log final statistics
    logger.info(f"Processing completed! Final accuracy: {results['accuracy']:.4f}")
    logger.info(
        f"Success: {results['successful_tasks']}, "
        f"Skipped: {results['skipped_tasks']}, "
        f"Failed: {results['failed_tasks']}"
    )


def save_final_results(output_path: Path, results: dict, args):
    """Save final processing results and statistics"""
    # Save detailed results
    with open(output_path / "summary.json", "w") as f:
        json.dump(
            {
                "total_tasks": results["total_tasks"],
                "successful_tasks": results["successful_tasks"],
                "skipped_tasks": results["skipped_tasks"],
                "failed_tasks": results["failed_tasks"],
                "accuracy": results["accuracy"],
                "all_accuracies": results["all_accuracies"],
            },
            f,
            indent=2,
        )

    # Save error report if there are failures
    failed_results = [r for r in results["results"] if r["status"] == "error"]
    if failed_results:
        with open(output_path / "error_report.json", "w") as f:
            json.dump(failed_results, f, indent=2)

    # Legacy stats file for compatibility
    with open(f"{output_path}_stat.txt", "w") as f:
        f.write(f"acc for {Path(args.data_path).stem}: {results['accuracy']}\n\n")
        f.write(f"All acc: {results['all_accuracies']}")


if __name__ == "__main__":
    main()
