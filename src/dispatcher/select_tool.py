"""
Chart Analysis Tool Selection Module

This module provides functionality to analyze chart images and automatically select
the most appropriate tools for answering questions about the charts.
"""

import os
import re
import json
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional

from tqdm import tqdm
from src.utils.tool_utils import encode_image
from openai import AzureOpenAI, OpenAI

# Configuration constants
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRY = 3
DEFAULT_RETRY_DELAY = 1.0


def get_system_prompt() -> str:
    """
    Generate the system prompt for tool selection.

    Returns:
        str: The formatted system prompt
    """
    return """You are an expert chart analyst specializing in automated tool selection for chart analysis tasks.

## Task Overview
Given a chart image and a related question, you must analyze both inputs and select all possible tools in the following that can help answer the question.

## Available Tools

### 1. zoom_in_one_subfigure
- Description: A tool to perform information extraction, pattern recognition, or value comparison on the provided image. You can use this tool to analyze the image and answer the question based on the image content. This tool can be used to analyze either the original or a processed (edited) image.
- Constraints: Your question should be sufficiently specific and relevant to the image content.

### 2. zoom_in_specific_region_via_xy_axis_for_number_counting  
- Description: A tool to zoom in the local region of the image for number counting with the specific range of x/y axis values. You can input either x_start/x_end or y_start/y_end to obtain the region to be zoomed in. You can use a broader range of x/y values to obtain a larger region to be zoomed in.
- Constraints: This tool does not support the images that contain multiple subcharts.

### 3. compile_code
- Description: A tool to execute the python code for numerical calculation and return the result. You can use this tool to execute the python code for numerical calculation and return the result by passing the code as a string. The code should be a valid python code that can be executed. The final result should be printed in the code.
- Constraints: The code should be a valid python code that can be executed. The code should not contain any input/file/plotting operations.

### 4. add_axvline
- Description: Adds a vertical (or horizontal) reference line on a chart. The tool locates the specified axis value and draws a line in the chosen color (default: red) on either the x-axis or y-axis. This tool can be used to add a reference line on the chart for better analysis in the next steps, such as comparing values with a specific threshold.
- Constraints: This tool does not support images that contain multiple subcharts.

### 5. mask_irrelevant_data_using_legend
- Description: Masks every data items **except** the specified ones according to the legend colors so that analysis focuses only on the specified data in the chart.
- Constraints: Select this tool if there is more than 3 data items in the chart and the question focuses on a single data item or any subset of data items (irrelevant with the others) identifiable by legend colors. Do not select if there are only 2 data items in the chart and you can clearly distinguish them.

### 6. extract_information
- Description: Extract information from chart images


## Selection Guidelines

1. **Mandatory Requirement**: Always include `extract_information` in your tool selection
2. **Quantity Limit**: Select maximum 5 tools per question
3. **Priority Order**: Consider tools in order of their relevance to the question type
4. **Empty Selection**: Return empty array [] if no tools are suitable
5. **Context Awareness**: Consider the "Description" and "Constraints" of each tool carefully

## Analysis Framework

1. **Question Analysis**: Identify the core task (counting, comparison, extraction, calculation)
2. **Chart Assessment**: Determine chart structure (single/multi-chart, complexity)
3. **Tool Matching**: Map identified needs to available tools
4. **Validation**: Ensure selected tools satisfy constraints and requirements

## Output Format

```json
{
    "analysis": "Detailed analysis of the question and chart, explaining the reasoning behind tool selection. Include chart type assessment, task identification, and tool applicability.",
    "tools": ["tool_name_1", "tool_name_2", ...]
}
```
"""


def setup_logger(model_name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Args:
        model_name: Name of the model for logging identification
        log_level: Logging level (default: INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"{model_name}_tool_selection")
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{model_name}_tool_selection.log"), encoding="utf-8"
    )
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def create_openai_client(
    model_name: str, base_url: str = "http://localhost:8000/v1"
) -> tuple[OpenAI, str]:
    """
    Create an OpenAI client for custom endpoints.

    Args:
        model_name: Name of the model to use
        base_url: Base URL for the API endpoint

    Returns:
        tuple: (OpenAI client, model_name)
    """
    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    return client, model_name


class ChartAnalysisGPT:
    """
    GPT client for chart analysis and tool selection tasks.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Initialize the GPT client.

        Args:
            max_tokens: Maximum tokens in response
            temperature: Temperature for response generation
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

    async def generate_response_async(
        self,
        client: AzureOpenAI,
        deployment_name: str,
        message: List[Dict[str, Any]],
        task_id: str,
        max_retry: int = DEFAULT_MAX_RETRY,
    ) -> tuple[str, str]:
        """
        Generate response asynchronously with retry logic.

        Args:
            client: Azure OpenAI client
            deployment_name: Model deployment name
            message: Message payload for the API
            task_id: Unique identifier for the task
            max_retry: Maximum number of retry attempts

        Returns:
            tuple: (task_id, response_content)
        """
        for attempt in range(max_retry):
            try:
                self.logger.info(
                    f"Generating response for task {task_id}, attempt {attempt + 1}/{max_retry}"
                )

                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=deployment_name,
                    messages=message,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                extracted_response = response.choices[0].message.content
                self.logger.info(f"Successfully generated response for task {task_id}")
                return task_id, extracted_response

            except Exception as e:
                self.logger.error(
                    f"Error for task {task_id}, attempt {attempt + 1}/{max_retry}: {e}"
                )
                if attempt == max_retry - 1:
                    self.logger.error(f"All retry attempts failed for task {task_id}")
                    return (
                        task_id,
                        f"Failed to generate response after {max_retry} attempts: {str(e)}",
                    )

                # Exponential backoff
                await asyncio.sleep(DEFAULT_RETRY_DELAY * (2**attempt))


def parse_tool_selection_response(response: str) -> Dict[str, Any]:
    """
    Parse the GPT response to extract tool selection results.

    Args:
        response: Raw response from GPT

    Returns:
        dict: Parsed response with analysis and selected tools
    """
    if not response:
        return {"analysis": "No response received", "tools": []}

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        json_content = json_match.group(1).strip()
        try:
            parsed_response = json.loads(json_content)
            # Validate required fields
            if "analysis" not in parsed_response:
                parsed_response["analysis"] = "Analysis not provided"
            if "tools" not in parsed_response:
                parsed_response["tools"] = []
            print(f"Parsed response: {parsed_response}")
            return parsed_response
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e} in response: {json_content}")
            return {"analysis": f"JSON parsing error: {str(e)}", "tools": []}

    # Fallback: try to parse the entire response as JSON
    try:
        parsed_response = json.loads(response)
        if "analysis" not in parsed_response:
            parsed_response["analysis"] = "Analysis not provided"
        if "tools" not in parsed_response:
            parsed_response["tools"] = []
        return parsed_response
    except json.JSONDecodeError:
        return {"analysis": "Unable to parse response", "tools": []}


async def process_data(
    data: Dict[str, Any],
    max_concurrent: int = 5,
    max_items: Optional[int] = None,
    image_dir: Optional[str] = None,
    client: Optional[AzureOpenAI] = None,
    deployment_name: Optional[str] = None,
    answer_option: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Process chart analysis data using GPT for tool selection.

    Args:
        data: Dictionary containing chart data
        max_concurrent: Maximum concurrent requests
        max_items: Maximum number of items to process
        image_dir: Directory containing chart images
        client: Azure OpenAI client
        deployment_name: Model deployment name
        answer_option: Answer generation option
        logger: Logger instance

    Returns:
        dict: Processing results
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(
        f"Starting data processing with max_concurrent={max_concurrent}, max_items={max_items}"
    )

    gpt_client = ChartAnalysisGPT()
    tasks = []
    results = {}

    # Prepare all tasks
    for task_id, item in data.items():
        if max_items and len(tasks) >= max_items:
            logger.info(
                f"Reached maximum number of items ({max_items}), stopping task creation"
            )
            break

        if not image_dir:
            image = item["figure_path"]
        else:
            image = os.path.join(image_dir, item["figure_path"])
        prompt = f"""## Question: {item["query"]}
Analyze the provided chart image and question, then select the most appropriate tools following the guidelines above.
"""

        if answer_option == "cot":
            prompt = (
                prompt
                + "\n\nAnalyze the chart step by step and then answer the question."
            )

        if not os.path.exists(image):
            logger.warning(f"Index {task_id}, image {image} does not exist.")
            continue

        base64_image = encode_image(image)
        system_prompt = get_system_prompt()
        message = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        task = {
            "id": task_id,
            "message": message,
            "query": item["query"],
            "prompt": prompt,
            "ground_truth": item["answer"],
        }
        tasks.append(task)

    logger.info(f"Created {len(tasks)} tasks to process")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_task(task):
        async with semaphore:
            logger.info(f"Processing task ID: {task['id']}")
            task_id, response = await gpt_client.generate_response_async(
                client, deployment_name, task["message"], task["id"]
            )
            logger.info(f"Completed task ID: {task_id}")
            return {
                "id": task_id,
                "query": task["query"],
                "ground_truth": task["ground_truth"],
                **parse_tool_selection_response(response),
            }

    # Create all async tasks and wait for completion
    running_tasks = [process_task(task) for task in tasks]
    completed_tasks = []

    logger.info(f"Starting to process {len(running_tasks)} tasks")
    for f in tqdm(asyncio.as_completed(running_tasks), total=len(running_tasks)):
        result = await f
        completed_tasks.append(result)

        task_id = result["id"]
        results[task_id] = result

    logger.info(f"Completed processing {len(completed_tasks)} tasks")
    return results


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Chart Analysis Tool Selection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/chartqapro_metadata.json",
        help="Path to the input data file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="out/tool_selection/chartqapro_gpt-4.1-mini_tool_selection.json",
        help="Path to save the output results",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/ChartQAPro/images",
        help="Directory containing chart images",
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4.1-mini",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--model_version", type=str, default="2025-04-14", help="Version of the model"
    )

    # Performance arguments
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=4,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of items to process (for testing)",
    )

    # Answer options
    parser.add_argument(
        "--answer_option",
        type=str,
        default="direct",
        choices=["direct", "cot"],
        help="Answer generation option",
    )

    return parser


def main():
    """Main function to run the tool selection system."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(args.model_name)

    try:
        # Initialize Azure client
        client, deployment_name = create_openai_client(
            args.model_name, base_url=os.environ.get("BASE_URL", "http://localhost:8000/v1")
        )

        # Load data
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data)} items from {args.data_path}")

        # Process data
        logger.info(
            f"Running with max_concurrent_requests={args.max_concurrent_requests}"
        )
        results = asyncio.run(
            process_data(
                data=data,
                max_concurrent=args.max_concurrent_requests,
                max_items=args.max_items,
                image_dir=args.image_dir,
                client=client,
                deployment_name=deployment_name,
                answer_option=args.answer_option,
                logger=logger,
            )
        )
        logger.info(f"Processing completed, got {len(results)} results")

        # Save results
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # sort results by task ID
        results = dict(sorted(results.items(), key=lambda item: int(item[0])))
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
