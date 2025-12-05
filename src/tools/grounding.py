import re
import os
import json
from openai import OpenAI
from qwen_vl_utils import smart_resize
from PIL import Image
from typing import Tuple, List
from src.utils.tool_utils import encode_image, values_to_pixel
from src.tools.answer_question import answer_question
from src.utils.model import ModelClient


class GroundingBboxDetector:
    def __init__(self, port=8000, model_name="GroundingModel"):
        if "gpt" in model_name:
            self.client = ModelClient().get_client()
        else:
            self.client = OpenAI(
                base_url=f"http://localhost:{port}/v1",
                api_key="local",
            )
        self.model_name = model_name
        self.ref_map = {
            "all": "all subfigures",
            "row": "row {row}",
            "col": "column {col}",
            "subfigure": "the subfigure at row {row}, column {col}",
            "title": "the chart title",
            "x_label": "the x-axis label",
            "y_label": "the y-axis label",
            "legend": "the legend",
            "legend_items": "all legend items",
            "axes": "the axes",
            "x_axis": "the x-axis",
            "y_axis": "the y-axis",
            "x": "on the x-axis where x = {value}",
            "y": "on the y-axis where y = {value}",
            "legend_item": "legend item {value}",
        }
        self.axis_map = {}
        self.bbox = None
        self.element_detector = self


    def _build_messages(self, image, ref_type="all", row=0, col=0, value=0):
        system_prompt = "You are a helpful assistant specialized in chart analysis. Examine the provided chart image and return the pixel coordinates of the requested element, or return 'Not found' if the element does not exist in the image."
        ref_str = self.ref_map.get(ref_type, ref_type).format(row=row, col=col, value=value)
        query = f"This is a chart image. Please locate the coordinates of <|object_ref_start|>{ref_str}<|object_ref_end|>."
        if ref_type in ["x", "y"]:
            query = f"Please locate the point <|object_ref_start|>{ref_str}<|object_ref_end|>. Give me the exact pixel position (x, y) in the image, where (0, 0) is the top-left corner of the image."
        if "gpt" in self.model_name:
            query += f"Identify the accurate pixel bounding box [x_min, y_min, x_max, y_max] of the region given the image with width:{image.width} and height:{image.height}."
            system_prompt = "You are a precise chart-grounding assistant. Your role is to carefully analyze chart images and provide accurate pixel-level bounding boxes for regions that match a given description."

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_image(image)}},
                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            },
        ]
        return messages

    def generate(self, messages, temperature=1e-6, max_tokens=512, seed=42):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        content = response.choices[0].message.content
        return content

    def resize_bbox(
        self, bbox: List[int], scale_factor: Tuple[float, float]
    ) -> List[int]:
        """resize bbox according to scale_factor"""
        if len(bbox) == 2:
            x, y = bbox
            new_x = int(x / scale_factor[0])
            new_y = int(y / scale_factor[1])
            return [new_x, new_y]
        x1, y1, x2, y2 = bbox
        new_x1 = int(x1 / scale_factor[0])
        new_y1 = int(y1 / scale_factor[1])
        new_x2 = int(x2 / scale_factor[0])
        new_y2 = int(y2 / scale_factor[1])
        return [new_x1, new_y1, new_x2, new_y2]

    def extract_answer(self, text, return_all=False, scale_factor=(1, 1)):
        def match_answer(text, pattern):
            matches = re.findall(pattern, text)
            bbox = []
            for match_ in matches:
                bbox.append(self.resize_bbox(list(map(int, match_)), scale_factor))
            return bbox
        
        def extract_ref_objects(text, patterns):
            results = {}
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if len(match) == 5:
                        ref_obj = match[0].strip()
                        coords = list(map(int, match[1:]))
                        results[ref_obj] = self.resize_bbox(coords, scale_factor)
            return results

        if isinstance(text, list):
            if len(text) > 0:
                text = text[0]
            else:
                return [] if return_all else {}
                
        # extract reference objects first
        ref_patterns = [
            r'legend item \((.*?)\): \((\d+),(\d+)\),\((\d+),(\d+)\)',
            r'legend item "(.*?)".+?: \((\d+),(\d+)\),\((\d+),(\d+)\)'
        ]
        
        ref_objects = extract_ref_objects(text, ref_patterns)
        if ref_objects:
            return ref_objects
            
        # then extract bounding boxes
        patterns = [
            r"<\|box_start\|>\(([-]?\d+),([-]?\d+)\),\(([-]?\d+),([-]?\d+)\)<\|box_end\|>",
            r"\(([-]?\d+),([-]?\d+)\),\(([-]?\d+),([-]?\d+)\)",
            r"\(([-]?\d+),([-]?\d+)\)",
            r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
            r"BBOX:\s*\[\s*([-]?\d+)\s*,\s*([-]?\d+)\s*,\s*([-]?\d+)\s*,\s*([-]?\d+)\s*\]",
            r"\[\s*([-]?\d+)\s*,\s*([-]?\d+)\s*,\s*([-]?\d+)\s*,\s*([-]?\d+)\s*\]",
        ]
        
        for pattern in patterns:
            bbox = match_answer(text, pattern)
            if bbox:
                if return_all:
                    # merge ref_objects if exist
                    if ref_objects:
                        return {"boxes": bbox, "ref_objects": ref_objects}
                    return bbox
                else:
                    return bbox[0]
                    
        # if no bbox found, return ref_objects if exist
        if ref_objects:
            if return_all:
                return {"ref_objects": ref_objects}
            else:
                # return the first ref_object bbox
                return next(iter(ref_objects.values())) if ref_objects else []
                
        return []
    
    def _resize_image(self, image_path, img_min_size=512 * 512, img_max_size=1280 * 28 * 28):
        image = Image.open(image_path) if isinstance(image_path, str) else image_path
        orig_width, orig_height = image.size
        h, w = smart_resize(
            height=orig_height,
            width=orig_width,
            min_pixels=img_min_size,
            max_pixels=img_max_size,
        )
        image = image.resize((w, h), Image.LANCZOS).convert("RGB")
        scale_factor = (w / orig_width, h / orig_height)
        return image, scale_factor

    def detect(self, image_path, ref_type="subfigure", row=0, col=0, value=0):
        """main function to detect bounding box"""
        self.image_path = image_path
        self.raw_image = Image.open(image_path) if isinstance(image_path, str) else image_path
        resized_image, scale_factor = self._resize_image(image_path)
        messages = self._build_messages(resized_image, ref_type, row, col, value)
        output_text = self.generate(messages)

        self.bbox = self.extract_answer(output_text, scale_factor=scale_factor)
        return self.bbox

    def detect_geometry_points(self, image_path: str, output_path: str) -> str:
        """
        Detect all point coordinates in the chart image.
        Returns a string representation of the coordinates.
        """
        self.image_path = image_path
        resized_image, _ = self._resize_image(image_path)
        resized_image.save(output_path)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in geometry detection.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image(resized_image)},
                    },
                    {
                        "type": "text",
                        "text": "Identify all points and their corresponding text labels with precise coordinate mapping.",
                    },
                ],
            },
        ]
        output_text = self.generate(messages)
        return output_text

    def detect_element_bbox(
        self,
        image_path: str,
        row: int = 0,
        col: int = 0,
        value: int = 0,
        ref_type="legend",
        min_img_size=512 * 512,
        max_img_size=1280 * 1280,
        output_path=None,
    ) -> List[List[int]]:
        resized_image, scale_factor = self._resize_image(image_path, min_img_size, max_img_size)
        messages = self._build_messages(
            resized_image, ref_type=ref_type, row=row, col=col, value=value
        )
        outputs = self.generate(messages)
        if output_path:
            resized_image.save(output_path)
        return self.extract_answer(outputs, return_all=True, scale_factor=scale_factor)

    def visualize(self, output_path):
        """visualize the detected bounding box on the image and save the cropped image"""
        if not self.bbox:
            print("No bounding box detected.")
            return
        
        width, height = self.raw_image.size
        x1, y1, x2, y2 = self.bbox
    
        if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            print(f"invalid coords: {self.bbox}")
            return self.bbox

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, width)
        y2 = min(y2, height)
        if x1 >= x2 or y1 >= y2:
            print(f"invalid clipped coords: {(x1, y1, x2, y2)}")
            return self.bbox

        try:
            # Perform cropping operation
            cropped_img = self.raw_image.crop((x1, y1, x2, y2))
        except Exception as e:
            raise RuntimeError(f"Image cropping failed: {str(e)}")

        # Save image if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                cropped_img.save(output_path)
                print(f"Saved cropped image to {output_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to save cropped image: {str(e)}")

        return (x1, y1, x2, y2)

    def get_point_coordinates(
        self, image_path: str, values: List[str], axis: str = "x"
    ) -> Tuple[int, int]:
        """
        Get the pixel coordinates of specific values on the x or y axis of a chart image.
        """
        assert axis in ["x", "y"], "Axis must be 'x' or 'y'."

        if image_path in self.axis_map and axis in self.axis_map[image_path]:
            # cached axis mapping
            axis_map = self.axis_map[image_path][axis]
            return values_to_pixel(
                values, axis_map["coff"], axis_map["bias"], axis, axis_map["avg_sec"]
            )

        prompt = f"""The image is a chart image. Output the second and third {axis}-axis values of the chart image.
        Output format:
        {{ "second_value": "value", "third_value": "value" }}
        """
        try:
            response = answer_question(question=prompt, image_path=image_path)
            pattern = r"```json\s*(.*?)\s*```"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                response = match.group(1).strip()
            response = json.loads(response)
            axis_min = float(response["second_value"])
            axis_max = float(response["third_value"])
        except Exception:
            # default values if parsing fails
            pixel_values = []
            for value in values:
                coordinates = self.detect_element_bbox(
                    image_path=image_path, ref_type=axis, value=value
                )
                if not coordinates:
                    raise ValueError(f"Value '{value}' not found on the {axis}-axis.")
                pixel_values.append(coordinates[0])
            return pixel_values

        primary_idx = 0 if axis == "x" else 1
        axis_min_pixel = self.detect_element_bbox(
            image_path=image_path, ref_type=axis, value=axis_min
        )[0]
        axis_max_pixel = self.detect_element_bbox(
            image_path=image_path, ref_type=axis, value=axis_max
        )[0]

        axis_avg_sec = (
            axis_max_pixel[1 - primary_idx] + axis_min_pixel[1 - primary_idx]
        ) // 2
        axis_min_pixel = axis_min_pixel[primary_idx]
        axis_max_pixel = axis_max_pixel[primary_idx]

        axis_coff = (axis_max_pixel - axis_min_pixel) / (axis_max - axis_min)
        axis_bias = axis_min_pixel - axis_coff * axis_min

        # cache the axis mapping
        if image_path not in self.axis_map:
            self.axis_map[image_path] = {}
        self.axis_map[image_path][axis] = {
            "coff": axis_coff,
            "bias": axis_bias,
            "avg_sec": axis_avg_sec,
            "min": axis_min,
            "max": axis_max,
        }
        return values_to_pixel(values, axis_coff, axis_bias, axis, axis_avg_sec)


    def visualize_with_legend_if_no_legend(self, output_path: str) -> Image.Image:
        if not self.bbox:
            raise ValueError("No bounding box detected. Run detect() first.")

        # crop main image
        img_width, img_height = self.raw_image.size
        main_bbox = (
            max(self.bbox[0], 0),
            max(self.bbox[1], 0),
            min(self.bbox[2], img_width),
            min(self.bbox[3], img_height)
        )

        # check main_bbox validity
        if main_bbox[0] >= main_bbox[2] or main_bbox[1] >= main_bbox[3]:
            raise ValueError(f"Invalid main bounding box: {main_bbox}")
        main_image = self.raw_image.crop(main_bbox)

        legend_bbox = self.element_detector.detect_element_bbox(self.image_path)
        legend_bbox_crop = self.element_detector.detect_element_bbox(main_image)

        if not legend_bbox or legend_bbox_crop:
            legend_bbox = [[0, 0, 0, 0]]
        legend_bbox = tuple(legend_bbox[0])

        if not legend_bbox or len(legend_bbox) != 4:
            raise ValueError("Legend bbox must be a tuple of 4 integers (x1, y1, x2, y2).")

        # check legend_bbox validity and crop within image bounds
        legend_bbox = (
            max(legend_bbox[0], 0),
            max(legend_bbox[1], 0),
            min(legend_bbox[2], img_width),
            min(legend_bbox[3], img_height)
        )
        if legend_bbox[0] > legend_bbox[2] or legend_bbox[1] > legend_bbox[3]:
            raise ValueError(f"Invalid legend bounding box: {legend_bbox}")
        legend_image = self.raw_image.crop(legend_bbox)

        if legend_image.size == (0, 0):
            combined_image = main_image
        elif legend_image.width / legend_image.height > 1:
            combined_width = max(main_image.width, legend_image.width)
            if legend_image.width > main_image.width:
                radio = combined_width / main_image.width
                main_image = main_image.resize((int(combined_width), int(main_image.height * radio)), Image.LANCZOS)
            else:
                radio = combined_width / legend_image.width
                legend_image = legend_image.resize((int(combined_width), int(legend_image.height * radio)), Image.LANCZOS)
            max_height = main_image.height + legend_image.height
            combined_image = Image.new('RGB', (combined_width, max_height), (255, 255, 255))
            combined_image.paste(legend_image, (0, 0))
            combined_image.paste(main_image, (0, legend_image.height))
        else:
            combined_height = max(main_image.height, legend_image.height)
            if legend_image.height > main_image.height:
                radio = combined_height / main_image.height
                main_image = main_image.resize((int(main_image.width * radio), int(combined_height)), Image.LANCZOS)
            else:
                radio = combined_height / legend_image.height
                legend_image = legend_image.resize((int(legend_image.width * radio), int(combined_height)), Image.LANCZOS)
            max_width = main_image.width + legend_image.width
            combined_image = Image.new('RGB', (max_width, combined_height), (255, 255, 255))
            combined_image.paste(main_image, (0, 0))
            combined_image.paste(legend_image, (main_image.width, 0))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_image.save(output_path)
        return main_bbox
