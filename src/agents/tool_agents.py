from typing import Dict, Callable, List, Any
import inspect
import functools
import re
import json
import math
import ast
import os

from pathlib import Path
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from src.tools import (
    GroundingBboxDetector,
    add_axvline,
    execute_code,
    answer_question,
    mask_chart_legend,
    denoise_bm3d_luma,
)
from src.critics.visual_critic import VisualCritic
from src.agents.reasoner import Reasoner


class ToolAgent:
    _tool_registry = {}

    def __init__(self):
        self._active_tools = {}

    @classmethod
    def register_tool(cls, name: str, param_schema: Dict = None):
        """register a tool method with optional parameter schema."""

        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(self, params: Dict) -> Dict:
                return func(self, params)

            # save method signature for parameter validation
            sig = inspect.signature(func)
            param_info = {
                p.name: {
                    "type": p.annotation,
                    "required": p.default == inspect.Parameter.empty,
                }
                for p in sig.parameters.values()
                if p.name != "self" and p.name != "params"
            }

            cls._tool_registry[name.upper()] = {
                "func": func,
                "param_schema": param_schema or param_info,
            }
            return wrapper

        return decorator

    def _has_tool(self, function_call: str) -> bool:
        """check if the function call matches any registered tool name"""
        for tool_name in self._tool_registry.keys():
            if tool_name in function_call.upper():
                return True
        return False

    def execute_action(
        self,
        action: str,
        question: str = "",
        instruction: str = "",
        tool_call_history: list = [],
        image_path_description: dict = {},
    ) -> str:
        try:
            # extract code block if exists
            if "```" in action:
                code_block_match = re.search(
                    r"```(?:python)?\n(.*?)```", action, re.DOTALL
                )
                if not code_block_match:
                    return "❌ Invalid code block format", None, None, tool_call_history, image_path_description, False

                full_call = code_block_match.group(1).strip()
            else:
                full_call = action
            tool_call_history.append(full_call)

            if not self._has_tool(full_call):
                logger.warning(f"Tool not found for call: {full_call}")
                tool_name = "COMPILE_CODE"
                raw_params = {"code": full_call}
            else:
                tool_name, raw_params = self._parse_function_call(full_call)
                if not tool_name:
                    return (
                        f"❌ Invalid tool call format: {full_call}",
                        None,
                        None,
                        tool_call_history,
                        image_path_description,
                        False,
                    )

            params = {}
            for key, value in raw_params.items():
                if key.endswith(("_path", "_dir")):
                    params[key] = str(Path(value).resolve())
                else:
                    try:
                        params[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        params[key] = value

            # execute the tool
            params["instruction"] = instruction
            if "question" in params:
                if params["question"] in question:
                    params["question"] = question
            if "image_path" in params:
                if params["image_path"] in image_path_description:
                    logger.info(
                        f"Image description: {image_path_description[params['image_path']]}"
                    )
                    params["description"] = image_path_description[params["image_path"]]
                if "output_path" in params and params["output_path"] == params["image_path"]:
                    params["output_path"] = str(
                        Path(params["image_path"]).parent
                        / f"processed_{Path(params['image_path']).name}"
                    )

            result = self._execute_tool(tool_name, params)

            model_response = None
            if "model_response" in result:
                model_response = result["model_response"]
            if "output_path" not in result:
                result["output_path"] = None
            if (
                "goal" in result
                and result["output_path"] is not None
            ):
                image_path_description[result["output_path"]] = result[
                    "goal"
                ]
            if (
                result["output_path"] is not None
                and "output_text" in result and os.path.exists(result["output_path"])
            ):
                output_path = result["output_path"]
                image_path_description[output_path] = result["output_text"]
            else:
                output_path = None
            return (
                self._format_tool_result(result),
                output_path,
                model_response,
                tool_call_history,
                image_path_description,
                True,
            )

        except Exception as e:
            return (
                f"❌ Execution error: {str(e)}",
                None,
                None,
                tool_call_history,
                image_path_description,
            )

    def _parse_function_call(self, code_block: str) -> tuple:
        """parse the function call from the code block"""
        cleaned_lines = []
        for line in code_block.splitlines():
            line = re.sub(r"^#.*", "", line).strip()  # remove full-line comments
            if line:
                cleaned_lines.append(line)
        if not cleaned_lines:
            return None, None

        combined = " ".join(cleaned_lines)

        if "(" not in combined or ")" not in combined:
            return None, None

        func_part, param_part = combined.split("(", 1)
        func_name = func_part.strip().upper()
        param_str = param_part.rsplit(")", 1)[0].strip()

        # param parsing with enhanced handling
        stack = []
        in_quotes = False
        quote_char = None
        current_key = None
        buffer = []
        params = {}

        def finalize_param():
            nonlocal current_key, buffer
            if not current_key and not buffer:
                return

            param_str = "".join(buffer).strip()
            if "=" in param_str and not current_key:
                key_part, _, value_part = param_str.partition("=")
                current_key = key_part.strip()
                param_str = value_part.strip()

            try:
                value = ast.literal_eval(param_str)
            except Exception:
                value = param_str
                logger.warning(
                    f"Parameter value '{param_str}' could not be converted. Using string."
                )

            if current_key and current_key.endswith(("_path", "_dir")):
                value = str(Path(value).resolve()) if isinstance(value, str) else value

            if current_key:
                params[current_key] = value
            elif param_str:
                params[f"arg{len(params) + 1}"] = value

            current_key = None
            buffer = []

        for char in param_str + ",":
            if char in ('"', "'"):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                buffer.append(char)
            elif in_quotes:
                buffer.append(char)
            else:
                if char in ("(", "[", "{"):
                    stack.append(char)
                elif char in (")", "]", "}"):
                    if stack:
                        stack.pop()

                if char == "," and not stack and not in_quotes:
                    finalize_param()
                elif char == "=" and not stack and not current_key:
                    current_key = "".join(buffer).strip()
                    buffer = []
                else:
                    buffer.append(char)

        finalize_param()
        return func_name, params

    def _format_success_result(self, data: Any) -> str:
        """Format tool execution results."""
        if isinstance(data, dict):
            items = []
            for k, v in data.items():
                if k == "bboxes":
                    items.append(f"Detect {len(v)} bboxes:")
                elif k == "output_path":
                    if v is not None and v != "None" and v != "":
                        items.append(f"The processed image is saved in: {v}")
                elif k == "output_text":
                    items.append(f"Output text: {v}")
                elif k == "compile_result":
                    items.append(f"Compile result: {v}")
            return "\n".join(items)
        return str(data)

    def _format_tool_result(self, result: Dict) -> str:
        """Unified formatting of tool execution results"""
        if result["status"] == "success":
            if "result" in result:
                result = result["result"]
            return self._format_success_result(result)
        else:
            result = result.get("output_text", "grounding failed.")
            return f"❌ Tool execution failed: {result}"

    def _execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """unified tool execution entry"""
        tool_name = tool_name.upper()
        if tool_name not in self._tool_registry:
            return {"status": "error", "message": f"Tool {tool_name} not registered"}

        try:
            # parameter validation
            validated = self._validate_params(
                params, self._tool_registry[tool_name]["param_schema"]
            )

            # execute tool function
            result = self._tool_registry[tool_name]["func"](self, validated)
            if "status" in result:
                return result
            return {"status": "success", "result": result}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _validate_params(self, params: Dict, schema: Dict) -> Dict:
        """Validate and convert parameters based on the provided schema."""
        validated = {}
        for param, config in schema.items():
            if param not in params:
                if config["required"]:
                    raise ValueError(f"Missing required parameter: {param}")
                elif "default" in config:
                    validated[param] = config["default"]
                continue

            value = params.get(param)
            if value is not None:
                # convert type if necessary
                try:
                    typed_value = config["type"](value)
                    validated[param] = typed_value
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid type for {param}: expected {config['type']}, got {type(value)}"
                    )

                # if the parameter has choices, validate that the value is within the allowed range
                if "choices" in config and typed_value not in config["choices"]:
                    raise ValueError(
                        f"Invalid value for {param}: {typed_value}. Must be one of {config['choices']}"
                    )

        for param, config in schema.items():
            if param not in validated and "default" in config:
                validated[param] = config["default"]

        return validated


class ImageTools(ToolAgent):
    def __init__(self, task_type: str = "chart"):
        self.bbox_detector = GroundingBboxDetector()
        self.visual_critic = VisualCritic()
        self.reasoner = Reasoner()
        self.task_type = task_type

    def _create_magnified_roi_image(
        self, image, roi_bbox, output_path, scale_factor=2.0
    ):
        # crop ROI region
        roi_image = image.crop(roi_bbox)

        # magnify ROI
        roi_width = roi_bbox[2] - roi_bbox[0]
        roi_height = roi_bbox[3] - roi_bbox[1]
        magnified_width = int(roi_width * scale_factor)
        magnified_height = int(roi_height * scale_factor)
        magnified_roi = roi_image.resize(
            (magnified_width, magnified_height), Image.LANCZOS
        )

        canvas_width = image.width + magnified_width + 50
        canvas_height = max(image.height, magnified_height + 50)
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        canvas.paste(image, (0, 0))

        roi_x = image.width + 25
        roi_y = (canvas_height - magnified_height) // 2

        # paste the magnified ROI onto the right side of the canvas
        canvas.paste(magnified_roi, (roi_x, roi_y))

        # create drawing object
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(roi_bbox, outline="red", width=2)
        magnified_bbox = (
            roi_x,
            roi_y,
            roi_x + magnified_width,
            roi_y + magnified_height,
        )
        draw.rectangle(magnified_bbox, outline="black", width=4)

        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except Exception as e:
            logger.warning(
                f"Failed to load truetype font: {str(e)}. Using default font."
            )
            font = ImageFont.load_default()
            font.size = 60

        text = "callout image"
        text_width = draw.textlength(text, font=font)
        text_x = roi_x + (magnified_width - text_width) // 2
        text_y = roi_y - 150

        draw.text((text_x, text_y), text, fill="black", font=font)

        roi_corners = [
            (roi_bbox[0], roi_bbox[1]),
            (roi_bbox[2], roi_bbox[1]),
            (roi_bbox[0], roi_bbox[3]),
            (roi_bbox[2], roi_bbox[3]),
        ]

        magnified_corners = [
            (roi_x, roi_y),
            (roi_x + magnified_width, roi_y),
            (roi_x, roi_y + magnified_height),
            (roi_x + magnified_width, roi_y + magnified_height),
        ]

        line_color = "black"
        line_width = 1
        for roi_corner, mag_corner in zip(roi_corners, magnified_corners):
            draw.line([roi_corner, mag_corner], fill=line_color, width=line_width)

        canvas.save(output_path)

        return output_path

    @ToolAgent.register_tool(
        name="ZOOM_IN_SPECIFIC_REGION_VIA_XY_AXIS_FOR_COUNTING",
        param_schema={
            "image_path": {"type": str, "required": True},
            "output_path": {"type": str, "required": True},
            "x_axis_start": {"type": str, "required": False, "default": None},
            "x_axis_end": {"type": str, "required": False, "default": None},
            "y_axis_start": {"type": str, "required": False, "default": None},
            "y_axis_end": {"type": str, "required": False, "default": None},
        },
    )
    def handle_extract_roi(self, params: Dict) -> Dict:
        """create a zoom-in callout image for the specified region"""
        image = Image.open(params["image_path"])
        image_width, image_height = image.size
        output_text = ""

        # Handle x coordinates
        if params["x_axis_start"] is not None and params["x_axis_end"] is not None:
            try:
                x_start = self.bbox_detector.get_point_coordinates(
                    params["image_path"], [params["x_axis_start"]], "x"
                )[0][0]
                x_end = self.bbox_detector.get_point_coordinates(
                    params["image_path"], [params["x_axis_end"]], "x"
                )[0][0]
            except Exception as e:
                return {
                    "status": "error",
                    "output_path": params["output_path"],
                    "output_text": f"Failed to detect x coordinates: {str(e)}",
                }
        else:
            x_start = 0
            x_end = image_width

        # Handle y coordinates
        if params["y_axis_start"] is not None and params["y_axis_end"] is not None:
            try:
                y_start = self.bbox_detector.get_point_coordinates(
                    params["image_path"], [params["y_axis_start"]], "y"
                )[0][1]
                y_end = self.bbox_detector.get_point_coordinates(
                    params["image_path"], [params["y_axis_end"]], "y"
                )[0][1]
            except Exception as e:
                return {
                    "status": "error",
                    "output_path": params["output_path"],
                    "output_text": f"Failed to detect y coordinates: {str(e)}",
                }
        else:
            y_start = image_height
            y_end = 0

        if y_start > y_end:
            y_start, y_end = y_end, y_start
        if x_start > x_end:
            x_start, x_end = x_end, x_start

        # Evaluate if coordinates are within chart range
        try:
            # Get chart bbox
            chart_bbox = self.bbox_detector.element_detector.detect_element_bbox(
                image_path=params["image_path"],
                ref_type="chart",
            )[0]
            chart_x_min, chart_y_min, chart_x_max, chart_y_max = chart_bbox

            # Check if coordinates are within chart range
            if x_start < chart_x_min or x_end > chart_x_max:
                x_start = max(x_start, chart_x_min)
                x_end = min(x_end, chart_x_max)
            if y_start < chart_y_min or y_end > chart_y_max:
                y_start = max(y_start, chart_y_min)
                y_end = min(y_end, chart_y_max)

        except Exception as e:
            return {
                "status": "error",
                "output_text": f"Failed to evaluate chart range: {str(e)}. Please use other tools for reasoning.",
            }

        try:
            bbox = (
                max(0, x_start),
                max(0, y_start),
                min(x_end, image.width),
                min(y_end, image.height),
            )

            ### assess the bbox area is large enough, at least 10% of the image area
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            image_area = image.width * image.height
            if bbox_area > 0.5 * image_area:
                logger.info("The bbox area is large enough, no need to zoom in.")
                return {
                    "status": "success",
                    "output_text": "The bbox area is too large, no need to zoom in.",
                }

            self._create_magnified_roi_image(image, bbox, params["output_path"])

            output_text += f"This is a callout or zoomed-in image of the region of interest (ROI) for the red bounding box from (x={params['x_axis_start']}, y={params['y_axis_start']}) to (x={params['x_axis_end']}, y={params['y_axis_end']}).\n"

            ### assess the bbox satisfies the goal
            goal = f"The red bbox is from (x={params['x_axis_start']}, y={params['y_axis_start']}) to (x={params['x_axis_end']}, y={params['y_axis_end']})."
            if "jpg" in params["output_path"]:
                assess_output_path = params["output_path"].replace(
                    ".jpg", "_assess.jpg"
                )
            elif "png" in params["output_path"]:
                assess_output_path = params["output_path"].replace(
                    ".png", "_assess.png"
                )
            else:
                raise ValueError(f"Unsupported image format: {params['output_path']}")

            assess_result, _ = self.visual_critic.assess_zoom_in(
                params["image_path"], bbox, assess_output_path, goal
            )
            if assess_result.lower() == "false":
                output_text = "This tool is not applicable to this image. Please try to use other tools for reasoning."
                return {"output_text": output_text}

            return {
                "status": "success",
                "output_text": output_text,
                "bbox": bbox,
                "output_path": params["output_path"],
            }
        except Exception as e:
            logger.error(f"Failed to create zoom-in image: {str(e)}")
            return {
                "status": "error",
                "output_text": f"Failed to create zoom-in image: {str(e)}",
            }

    @ToolAgent.register_tool(
        name="FOCUS_ON_SPECIFIC_DATA",
        param_schema={
            "image_path": {"type": str, "required": True},
            "output_path": {"type": str, "required": True},
            "mask_names_list": {"type": list, "required": True},
            "focused_names_list": {"type": list, "required": True},
            "extend_to_light_area": {"type": bool, "required": False, "default": False},
        },
    )
    def handle_element_mask(self, params: Dict) -> Dict:
        """mask irrelevant data based on legend items"""

        def add_color_mask(
            image_path,
            output_path,
            mask_legend_items,
            selected_legend_items,
            ref_type="legend_item",
        ):
            """add color mask to the chart based on legend items"""
            bboxes_to_mask = []
            bboxes_to_keep = []

            denoise_path = output_path.replace(".jpg", "_denoise.jpg").replace(
                ".png", "_denoise.png"
            )
            denoise_bm3d_luma(img_path=image_path, out_path=denoise_path)
            for legend in mask_legend_items:
                if legend.startswith("legend item"):
                    legend = legend.replace("legend item", "").strip()
                # get legend bbox
                legend_bbox = self.bbox_detector.element_detector.detect_element_bbox(
                    image_path=denoise_path,
                    ref_type=ref_type,
                    value=legend,
                )
                if legend_bbox:
                    bboxes_to_mask.append(legend_bbox[0])
                else:
                    raise ValueError(f"Legend '{legend}' not found in the image.")

            for legend in selected_legend_items:
                # get legend bbox
                if legend.startswith("legend item"):
                    legend = legend.replace("legend item", "").strip()
                legend_bbox = self.bbox_detector.element_detector.detect_element_bbox(
                    image_path=denoise_path,
                    ref_type=ref_type,
                    value=legend,
                )
                if legend_bbox:
                    bboxes_to_keep.append(legend_bbox[0])
                else:
                    raise ValueError(f"Legend '{legend}' not found in the image.")
            output_path = mask_chart_legend(
                image_path=image_path,
                output_path=output_path,
                mask_legend_bboxes=bboxes_to_mask,
                selected_legend_bboxes=bboxes_to_keep,
                extend_to_light_area=params.get("extend_to_light_area", False),
            )

        image_path = params["image_path"]
        output_path = params.get("output_path")
        mask_legend_items = params.get("mask_names_list")
        selected_legend_items = params.get("focused_names_list")

        try:
            add_color_mask(
                image_path=image_path,
                output_path=output_path,
                mask_legend_items=mask_legend_items,
                selected_legend_items=selected_legend_items,
            )
        except Exception as e:
            return {
                "status": "error",
                "output_text": f"Failed to mask irrelevant data: {str(e)}",
                "output_path": output_path,
            }
        output_text = (
            f"This image is focused on the selected data: {selected_legend_items}."
        )
        return {
            "status": "success",
            "output_text": output_text,
            "output_path": output_path,
        }

    @ToolAgent.register_tool(
        name="EXTRACT_INFORMATION",
        param_schema={
            "question": {"type": str, "required": True},
            "image_path": {"type": str, "required": True},
            "description": {"type": str, "required": False},
            "related_info": {"type": str, "required": False},
            "instruction": {"type": str, "required": False},
        },
    )
    def handle_extract_information(self, params: Dict) -> Dict:
        question = params["question"]
        image_path = params["image_path"]
        if self.task_type == "chart":
            self.visual_critic.assess_input(
                image_path, question, description=params.get("description", None)
            )
        try:
            if "description" in params:
                description = (
                    "Description of the chart: " + params["description"] + "\n"
                )
            else:
                description = ""

            sub_question_prompt = params["question"] + "\n"
            if "related_info" in params:
                related_info = f"(Related Information: {params['related_info']})\n"
            else:
                related_info = "\n"

            result = self.reasoner.reason(
                subquery=description + sub_question_prompt + related_info,
                image_path=params["image_path"],
            )
            return {
                "status": "success",
                "model_response": result,
                "output_path": params.get("output_path", None),
            }
        except Exception as e:
            return {
                "status": "error",
                "output_text": str(e),
                "output_path": params.get("output_path", None),
            }

    @ToolAgent.register_tool(
        name="COMPILE_CODE",
        param_schema={
            "code": {"type": str},
        },
    )
    def handle_code_compilation(self, params: Dict) -> Dict:
        try:
            # execute code compilation
            result = execute_code(params["code"])
            return {
                "compile_result": result,
                "output_path": params.get("output_path", None),
            }
        except Exception as e:
            return {
                "status": "error",
                "output_text": f"Code compilation failed: {str(e)}",
                "output_path": params.get("output_path", None),
            }

    @ToolAgent.register_tool(
        name="ZOOM_IN_ONE_SUBFIGURE",
        param_schema={
            "image_path": {"type": str, "required": True},
            "output_path": {"type": str, "required": True},
            "row": {"type": int, "required": True},
            "col": {"type": int, "required": True},
        },
    )
    def handle_subfigure_grounding(self, params: Dict) -> Dict:
        """zoom in one subfigure based on row and column indices"""
        try:
            # detect the bbox of the specified subfigure
            bbox = self.bbox_detector.detect(
                image_path=params["image_path"],
                row=params["row"],
                col=params["col"],
                ref_type="subfigure",
            )

            # check if bbox is valid
            if not bbox or len(bbox) != 4:
                return {
                    "status": "error",
                    # "message": "Failed to detect subfigure",
                    "output_path": params["output_path"],
                    "output_text": "Failed to detect subfigure. Please use the other tools for reasoning.",
                }

            # visualize the detected bbox
            bbox = self.bbox_detector.visualize_with_legend_if_no_legend(
                output_path=params["output_path"]
            )

            # return the result
            row = params["row"]
            col = params["col"]
            return {
                "status": "success",
                "output_text": f"This is the cropped subfigure at row {row} and {col} in the original image.",
                "output_path": params["output_path"],
                "bbox": bbox,
            }

        except Exception as e:
            return {
                "status": "error",
                "output_text": "The zoom_in_one_subfigure tool executed failed for this chart. Please use the other tools for reasoning.",
                "output_path": params["output_path"],
            }

    @ToolAgent.register_tool(
        name="ADD_AXVLINE",
        param_schema={
            "image_path": {"type": str, "required": True},
            "output_path": {"type": str, "required": True},
            "axis": {"type": str, "required": True, "choices": ["x", "y"]},
            "value": {"type": str, "required": True},
            "axvline_color": {"type": str, "required": False, "default": "red"},
        },
    )
    def handle_add_axvline(self, params: Dict) -> Dict:
        """add axvline on the chart"""
        image_path = params["image_path"]
        output_path = params["output_path"]
        axis = params["axis"]
        value = params["value"]
        color = params["axvline_color"]

        point_coords = self.bbox_detector.get_point_coordinates(
            image_path=image_path, values=[value], axis=axis
        )

        try:
            add_axvline(
                image_path=image_path,
                points=point_coords,
                output_path=output_path,
                axis=axis,
                color=color,
            )
        except Exception:
            return {
                "status": "error",
                "output_text": "Failed to add axvline on the chart. Please use other tools for reasoning.",
                "output_path": output_path,
            }
        return {
            "status": "success",
            "output_text": f"Added axvline at {value} on {axis} axis.",
            "output_path": output_path,
        }

    ####################################################################
    ### geometry tools
    ####################################################################
    def get_all_point_coordinates(
        self, image_path: str, output_path: str, points: List[str]
    ) -> Dict:
        """get all point coordinates via LLM reasoning"""
        coordinates = self.bbox_detector.detect_geometry_points(
            image_path=image_path,
            output_path=output_path,
        )
        prompt = f"""Given the point following coordinates, please help me extract the coordinates of each point.
Coordinates: {coordinates}
Target Points: {", ".join(points)}

Output format:
```json
{{
    "label_1": [x1, y1],
    "label_2": [x2, y2],
    ...
}}
```

"""
        point_coordinates = answer_question(
            question=prompt, system_prompt="You are a helpful assistent."
        )
        pattern = r"```json\n(.*?)\n```"
        match = re.search(pattern, point_coordinates, re.DOTALL)
        if match:
            try:
                point_coordinates = json.loads(match.group(1))
                for point in points:
                    if point not in point_coordinates:
                        raise ValueError(
                            f"Point '{point}' not found in the coordinates."
                        )
                return point_coordinates
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {str(e)}")
                return None
        else:
            return None

    @ToolAgent.register_tool(
        name="connect_points",
        param_schema={
            "image_path": {"type": str, "required": True},
            "output_path": {"type": str, "required": True},
            "start_point": {"type": str, "required": True},
            "end_point": {"type": str, "required": True},
        },
    )
    def handle_connect_points(self, params: Dict) -> Dict:
        """connect two points with a straight line"""
        try:
            start_point = params["start_point"]
            end_point = params["end_point"]

            coordinates = self.get_all_point_coordinates(
                image_path=params["image_path"],
                output_path=params["output_path"],
                points=[start_point, end_point],
            )
            if not coordinates:
                return {
                    "status": "error",
                    "message": "Failed to get coordinates for the specified points.",
                    "output_path": params["output_path"],
                }
            start_point = coordinates[start_point]
            end_point = coordinates[end_point]

            image = Image.open(params["output_path"])
            draw = ImageDraw.Draw(image)

            draw.line([start_point, end_point], fill="red", width=2)

            image = image.convert("RGB")
            image.save(params["output_path"])

            output_text = (
                f"Connected points {start_point} and {end_point} in the image."
            )
            return {"output_text": output_text, "output_path": params["output_path"]}
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "output_path": params["output_path"],
            }

    @ToolAgent.register_tool(
        name="draw_perpendicular_line",
        param_schema={
            "image_path": {"type": str, "required": True},
            "output_path": {"type": str, "required": True},
            "point": {"type": str, "required": True},
            "line_point1": {"type": str, "required": True},
            "line_point2": {"type": str, "required": True},
        },
    )
    def handle_draw_perpendicular_line(self, params: Dict) -> Dict:
        """draw perpendicular line from a point to a line defined by two points"""
        try:
            point = params["point"]
            line_point1 = params["line_point1"]
            line_point2 = params["line_point2"]

            coordinates = self.get_all_point_coordinates(
                image_path=params["image_path"],
                output_path=params["output_path"],
                points=[point, line_point1, line_point2],
            )
            if not coordinates:
                return {
                    "status": "error",
                    "message": "Failed to get coordinates for the specified points.",
                    "output_path": params["output_path"],
                }

            point = coordinates[point]
            line_point1 = coordinates[line_point1]
            line_point2 = coordinates[line_point2]

            dx = line_point2[0] - line_point1[0]
            dy = line_point2[1] - line_point1[1]
            line_len = math.sqrt(dx * dx + dy * dy)

            if line_len == 0:
                raise ValueError("Line points cannot be the same")

            px = point[0] - line_point1[0]
            py = point[1] - line_point1[1]
            dot_product = (px * dx + py * dy) / (line_len * line_len)

            proj_x = line_point1[0] + dot_product * dx
            proj_y = line_point1[1] + dot_product * dy
            projection_point = (int(proj_x), int(proj_y))

            image = Image.open(params["output_path"])
            draw = ImageDraw.Draw(image)

            draw.line([line_point1, line_point2], fill="gray", width=1)

            start_point = point
            end_point = projection_point
            if params.get("extend_beyond_projection", False):
                extend_dx = projection_point[0] - point[0]
                extend_dy = projection_point[1] - point[1]
                extend_len = math.sqrt(extend_dx * extend_dx + extend_dy * extend_dy)

                if extend_len > 0:
                    unit_extend_dx = extend_dx / extend_len
                    unit_extend_dy = extend_dy / extend_len
                    end_point = (
                        int(projection_point[0] + unit_extend_dx * extend_len),
                        int(projection_point[1] + unit_extend_dy * extend_len),
                    )

            draw.line([start_point, end_point], fill="red", width=2)

            draw.ellipse(
                [point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill="blue"
            )

            draw.ellipse(
                [
                    projection_point[0] - 3,
                    projection_point[1] - 3,
                    projection_point[0] + 3,
                    projection_point[1] + 3,
                ],
                fill="green",
            )
            draw.text(
                (projection_point[0] + 5, projection_point[1] - 5), "P", fill="green"
            )

            # label the extended end point if applicable
            if (
                params.get("extend_beyond_projection", False)
                and end_point != projection_point
            ):
                draw.ellipse(
                    [
                        end_point[0] - 3,
                        end_point[1] - 3,
                        end_point[0] + 3,
                        end_point[1] + 3,
                    ],
                    fill="orange",
                )
                draw.text((end_point[0] + 5, end_point[1] - 5), "E", fill="orange")

            image = image.convert("RGB")
            image.save(params["output_path"])

            distance = math.sqrt(
                (point[0] - projection_point[0]) ** 2
                + (point[1] - projection_point[1]) ** 2
            )
            output_text = f"Drew perpendicular from point {point} to its projection {projection_point} on line from {line_point1} to {line_point2}. Distance: {distance:.1f} pixels."

            return {
                "output_text": output_text,
                "output_path": params["output_path"],
                "projection_point": projection_point,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "output_path": params["output_path"],
            }

    @ToolAgent.register_tool(
        name="draw_parallel_line",
        param_schema={
            "image_path": {"type": str, "required": True},
            "output_path": {"type": str, "required": True},
            "point": {"type": str, "required": True},
            "line_point1": {"type": str, "required": True},
            "line_point2": {"type": str, "required": True},
            "line_length": {"type": int, "required": False, "default": 200},
        },
    )
    def handle_draw_parallel_line(self, params: Dict) -> Dict:
        """draw parallel line through a point to a line defined by two points"""
        try:
            point = params["point"]
            line_point1 = params["line_point1"]
            line_point2 = params["line_point2"]

            coordinates = self.get_all_point_coordinates(
                image_path=params["image_path"],
                output_path=params["output_path"],
                points=[point, line_point1, line_point2],
            )
            if not coordinates:
                return {
                    "status": "error",
                    "message": "Failed to get coordinates for the specified points.",
                    "output_path": params["output_path"],
                }

            point = coordinates[point]
            line_point1 = coordinates[line_point1]
            line_point2 = coordinates[line_point2]

            line_length = params.get("line_length", 400)
            dx = line_point2[0] - line_point1[0]
            dy = line_point2[1] - line_point1[1]

            if dx == 0 and dy == 0:
                raise ValueError("Line points cannot be the same")

            line_len = math.sqrt(dx * dx + dy * dy)

            unit_dx = dx / line_len
            unit_dy = dy / line_len
            half_length = line_length // 2
            start_point = (
                int(point[0] - unit_dx * half_length),
                int(point[1] - unit_dy * half_length),
            )
            end_point = (
                int(point[0] + unit_dx * half_length),
                int(point[1] + unit_dy * half_length),
            )

            image = Image.open(params["output_path"])
            draw = ImageDraw.Draw(image)

            draw.line([start_point, end_point], fill="red", width=2)

            draw.ellipse(
                [point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill="blue"
            )

            image = image.convert("RGB")
            image.save(params["output_path"])

            output_text = f"Drew parallel line through point {point} to line from {line_point1} to {line_point2}."
            return {"output_text": output_text, "output_path": params["output_path"]}
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "output_path": params["output_path"],
            }


if __name__ == "__main__":
    image_tool_agent = ImageTools()
    zoom_in_call = 'zoom_in_one_subfigure(image_path="data/CharXiv/images/0.jpg", output_path="output/zoom_in_subfigure.jpg", row=1, col=2)'
    result = image_tool_agent.execute_action(zoom_in_call)
    print(result)

