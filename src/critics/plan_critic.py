import json
import argparse
import os
from src.utils.model import ModelClient
import base64
from tqdm import tqdm
from PIL import Image
from src.utils.tool_utils import resize_max_edge
from concurrent.futures import ThreadPoolExecutor, as_completed


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


client = ModelClient().get_client()


def select_tool_with_gpt(figure_path, question, current_plan=None, tools=None):
    max_file_size_mb = -1
    max_image_pixels = 1024

    file_size_mb = os.path.getsize(figure_path) / (1024 * 1024)
    if max_file_size_mb > 0 and file_size_mb > max_file_size_mb:
        print(f"Image {figure_path} is too large ({file_size_mb:.2f} MB).")
        # resize the image and save to tmp directory
        img = Image.open(figure_path)
        img = resize_max_edge(img, max_size=max_image_pixels)
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        base_name = os.path.basename(figure_path)
        name, ext = os.path.splitext(base_name)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"
        save_path = os.path.join(tmp_dir, f"{name}_resized{ext}")

        if ext.lower() == ".png":
            img.save(save_path, "PNG", optimize=True)
        else:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(save_path, "JPEG", quality=85, optimize=True)

        resized_file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        if resized_file_size_mb > max_file_size_mb:
            raise ValueError(
                f"Resized image {save_path} still exceeds {max_file_size_mb} MB ({resized_file_size_mb:.2f} MB)."
            )

        print(
            f"Image resized from {file_size_mb:.2f} MB to {resized_file_size_mb:.2f} MB"
        )
        figure_path = save_path

    base64_image = encode_image(figure_path)
    prompt = f"""
You are a helpful AI assistant. You are given a question and a chart image. 
You need to think with the question and the chart image, and then evolve the plan to answer the question by adjusting the tools and the corresponding parameters.
If the current plan is perfect, you should maintain the current plan.

Available Tools:
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
- Description: Masks every data items **except** the specified ones according to the legend colors so that analysis focuses only on the specified data in the chart.\n🚨 MANDATORY: If the question mentions a single data item or any subset of data items identifiable by legend colors, **do not answer directly**—always apply this tool to mask irrelevant data before analysis.
- Constraints: This tool does not support the images that contain multiple subcharts. The legend items should be the text in the chart legend. The specified data cannot be differentiated with others by color, or there is no legend.

### 6. extract_information
**Purpose**: Extract information from chart images
**Parameters**:
- image_path (str): Absolute path to input image
- question (str): Question to be answered

*tool list: {tools}

*Question: {question}

*Reasoning process: {current_plan}

**Instructions:
1. You need to analyze the question and the answer to assess the correctness of the final answer based on the image.
2. Some intermediate information output from the tools may be incorrect, and you need to analyze the correctness of the extracted information.
3. If the final answer is correct, you should mark "ADJUSTMENT: False", else "ADJUSTMENT: True".
4. If the final answer is incorrect, you should analyze the incorrect reasoning process and the extracted information with mark "ADJUSTMENT: True".
5. If some tools failed or incorrectly applied to the image during the reasoning process, you must remove them from the tool list. The updated tool list is output as "tools: [tool1, tool2, tool3]".
6. If ADJUSTMENT is True, you should provide the detailed suggestions for the incorrect parts.


*Output Format:
Analysis: The strict analysis of the visual alignment of each output from the tools and the image. Step 1: Analyze the correctness of the extracted information. ... Step N: Analyze the correctness of the final answer.
ADJUSTMENT: True when the final answer is incorrect.
Suggestion: You should clearly state the suggestions about which tools, parameters, and the corresponding subquestions are used for the final answer. If ADJUSTMENT is False, output "Suggestion: None".
tool_analysis: The analysis of the tool using. If ADJUSTMENT is False, output "tool_analysis: None".
tools: [value_extraction, extract_information, tool1, tool2, tool3]. If ADJUSTMENT is False, output "tools: None". value_extraction and extract_information are default in the tool list.
        """
    message = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    try:
        response = client.generate(message)
        print("response: ", response)

        return response
    except Exception as e:
        print(f"Error extracting meta: {e}")
        return None


def process_single_item_for_tool_selection(id, data, image_dir):
    """for single item tool selection"""
    new_data = data.copy()
    question = data["query"]
    figure_path = os.path.join(image_dir, data["figure_path"])
    response = select_tool_with_gpt(figure_path, question)
    try:
        tools = response.split("Selected Tools:")[1].strip()
        new_data["tools"] = tools
    except Exception as e:
        print(f"Error processing tool selection for item {id}: {e}")
        new_data["tools"] = ""
    return id, new_data


def adjust_plan_single(
    data_path: str, image_dir: str, output_path: str, max_workers: int = 4
):
    """
    Select the tool for the data with concurrent execution
    """
    json_data = json.load(open(data_path, "r"))
    new_json_data = {}

    tasks = [(id, data, image_dir) for id, data in json_data.items()]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all tasks
        future_to_id = {
            executor.submit(process_single_item_for_tool_selection, *task): task[0]
            for task in tasks
        }

        with tqdm(total=len(tasks), desc="Selecting tools") as pbar:
            for future in as_completed(future_to_id):
                try:
                    id, new_data = future.result()
                    new_json_data[id] = new_data
                except Exception as e:
                    id = future_to_id[future]
                    print(f"Error processing tool selection for item {id}: {e}")
                    # set default empty tools for error items
                    new_data = json_data[id].copy()
                    new_data["tools"] = ""
                    new_json_data[id] = new_data
                finally:
                    pbar.update(1)

    json.dump(new_json_data, open(output_path, "w"), indent=4)


def process_single_item(id, data, response_dir, image_dir):
    """for single data item processing, used for concurrent execution"""
    new_data = data.copy()

    if not os.path.exists(os.path.join(response_dir, f"{id}/all_message.json")):
        print(f"Response for {id} does not exist, skipping...")
        new_data["adjustment"] = "False"
        return id, new_data, 0, 0  # total_increment, incorrect_increment

    question = data["query"]
    figure_path = os.path.join(image_dir, data["figure_path"])
    initial_acc = json.load(open(os.path.join(response_dir, f"{id}/answer.json"), "r"))[
        "acc"
    ]
    new_data["initial_acc"] = initial_acc
    original_output = json.load(
        open(os.path.join(response_dir, f"{id}/all_message.json"), "r")
    )
    current_plan = (
        original_output[0]["content"][1]["text"].split("Response History:")[1].strip()
    )
    response = select_tool_with_gpt(figure_path, question, current_plan, data["tools"])

    total_increment = 0
    incorrect_increment = 0

    try:
        adjustment = (
            response.split("ADJUSTMENT:")[1].strip().split("Suggestion:")[0].strip()
        )
        plan_reflection = (
            response.split("Analysis:")[1].strip().split("ADJUSTMENT:")[0].strip()
        )
        suggestion = (
            response.split("Suggestion:")[1].strip().split("tool_analysis:")[0].strip()
        )
        tool_analysis = (
            response.split("tool_analysis:")[1].strip().split("tools:")[0].strip()
        )
        tools = response.split("tools:")[1].strip()
        if "none" in tools.lower():
            tools = data["tools"]

        new_data["adjustment"] = adjustment
        new_data["plan_reflection"] = plan_reflection
        new_data["suggestion"] = suggestion
        new_data["tools"] = tools
        new_data["tool_analysis"] = tool_analysis

        if "true" in adjustment.lower():
            total_increment = 1
            if initial_acc == 0:
                incorrect_increment = 1

    except Exception as e:
        print(f"Error processing item {id}: {e}")
        new_data["adjustment"] = "False"
        new_data["plan_reflection"] = ""
        new_data["suggestion"] = "None"
        new_data["tools"] = data["tools"]
        new_data["tool_analysis"] = "None"

    print(
        f"ID: {id}, initial_acc: {initial_acc}, adjustment: {new_data.get('adjustment', 'False')}"
    )
    return id, new_data, total_increment, incorrect_increment


def adjust_plan(
    response_dir: str,
    output_path: str,
    data_path: str,
    image_dir: str,
    max_workers: int = 4,
):
    json_data = json.load(open(data_path, "r"))
    new_json_data = {}
    total_num = 0
    incorrect_num = 0

    tasks = [(id, data, response_dir, image_dir) for id, data in json_data.items()]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(process_single_item, *task): task[0] for task in tasks
        }

        with tqdm(total=len(tasks), desc="Processing items") as pbar:
            for future in as_completed(future_to_id):
                try:
                    id, new_data, total_increment, incorrect_increment = future.result()
                    new_json_data[id] = new_data
                    total_num += total_increment
                    incorrect_num += incorrect_increment
                except Exception as e:
                    id = future_to_id[future]
                    print(f"Error processing item {id}: {e}")
                    new_data = json_data[id].copy()
                    new_data["adjustment"] = "False"
                    new_data["plan_reflection"] = ""
                    new_data["suggestion"] = "None"
                    new_data["tools"] = json_data[id]["tools"]
                    new_data["tool_analysis"] = "None"
                    new_json_data[id] = new_data
                finally:
                    pbar.update(1)

    print("Total number of plans adjusted: ", total_num)
    print("Number of plans adjusted with incorrect initial accuracy: ", incorrect_num)
    json.dump(new_json_data, open(output_path, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="out/chartxiv_val_gpt_selection.json"
    )
    parser.add_argument(
        "--response_dir", type=str, default="out/gpt-4.1-mini_2025-04-14-charxiv_val"
    )
    parser.add_argument(
        "--output_path", type=str, default="out/chartxiv_val_gpt_selection_refined.json"
    )
    parser.add_argument("--image_dir", type=str, default="data")
    parser.add_argument("--model_version", type=str, default="gpt-4.1-mini_2025-04-14")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of concurrent workers",
    )
    args = parser.parse_args()
    os.environ["MODEL_VERSION"] = args.model_version
    os.environ["DEPLOYMENT_NAME"] = args.model_version
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    adjust_plan(
        args.response_dir,
        args.output_path,
        args.data_path,
        args.image_dir,
        args.max_workers,
    )
