import json
import re
from loguru import logger
from PIL import Image, ImageDraw

from src.utils.model import ModelClient
from src.prompts.manager import prompt_manager


def add_bbox_to_image(image_path, bbox, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=4)
    image.save(output_path)
    return output_path

class VisualCritic:
    def __init__(self):
        self.client = ModelClient()
        self.prompt = prompt_manager.get_prompt("visual_critic")

    def assess_zoom_in(self, image_path, bbox, output_path, goal=None):
        try:
            output_path = add_bbox_to_image(image_path, bbox, output_path)
            prompt = f"""The red bbox will be zoomed in on the chart image. Your task is to check if the red bbox strictly satisfies the goal. If yes, return true, otherwise return false with the reason and feedback to illustrate how to revise the bbox by increasing or decreasing the coordinates of the bbox.

Goal: {goal}

Output Format:
Reason: [Bbox analysis]
Result: [True/False]
Feedback: [Feedback]

Example Response:
Reason: The goal is to zoom in on the lower part of the chart where DAVIS-2017 J & F Mean score is less than 50. The red bbox is zoomed in on the lower part of the chart where DAVIS-2017 J & F Mean score is less than 40, which is not the goal.
Result: False
Feedback: The bbox is zoomed in on the lower part of the chart where DAVIS-2017 J & F Mean score is less than 40 instead of 50. You should decrease the coordinates of y1 of the bbox.
    """
            message = self.prompt.get_messages(
                images=output_path,
                critique_prompt=prompt,
            )

            response = self.client.generate(message)
            result = response.split("Result:")[1].split("Feedback:")[0].strip()
            feedback = response.split("Feedback:")[1].strip()
        except Exception as e:
            logger.error(f"Error parsing result: {e}")
            result = "False"
            feedback = e
        return result, feedback

    def assess_input(self, image_path, question, description=None):
        assess_prompt = f"""{description} Please check whether the given image contains enough information to answer the question: {question}. For example: 
    - If the question requires value extraction from a chart, the image should contain the chart with clear axis and data points.
    - If the question is about the trend of a chart, it needn't contain the chart title or axis labels.
    - If the question is about one subchart compared to another, it's acceptable to have only one subchart in the image.
    - If the image is extremely incomplete or the question is not related to the image, return false.
    - If the image is complete and the question is suitable for answering with the image, return true.

    Output Format:
    ```json
    {{
        "reason": "Provide a brief explanation of the assessment. For example, 'The image editing is failed because it's incomplete, lacking the legend.' or 'The image is complete, but the question is not suitable for directly answering with the image.'",
        "status": "true" or "false"
    }}
    ```

    """
        if "data" not in image_path:
            messages = self.prompt.get_messages(
                images=image_path,
                critique_prompt=assess_prompt,
            )
            assess_result = self.client.generate(messages)
            match_result = re.search(r"```json\n(.*?)\n```", assess_result, re.DOTALL)
            error_message = "This image does not contain enough information to answer the question, please use the original image."
            if match_result:
                assess_json = match_result.group(1)
                assess_data = json.loads(assess_json)
                logger.debug(f"Visual Critic Assess Result: {assess_data}")
                if assess_data["status"].lower() == "false":
                    raise ValueError(
                        f"The given image is not suitable for answering the question: {assess_data['reason']}\n"
                        f"Please check your previous tool calling steps. If previous steps are all correct, please use the original image."
                    )
            else:
                raise ValueError(error_message)

        else:
            logger.info("Skipping original image used, as image path contains 'data'.")
