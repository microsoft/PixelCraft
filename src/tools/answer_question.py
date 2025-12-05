from src.utils.model import ModelClient
import base64

SYSTEM_PROMPT = "You are an expert in chart analysis and data visualization. Please analyze the chart carefully and provide accurate answers based on the visual data presented."
GEO_SYSTEM_PROMPT = "You are an expert geometry problem solver. Given a geometry problem with an image, solve it step by step."

model_client = ModelClient()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def answer_question(question, image_path=None, system_prompt=None, task_type="chart"):
    """
    Directly answer the question based on the image.
    :param question: The question to be answered.
    :param image_path: The path to the image. Can be a string or a list of strings.
    :param system_prompt: The system prompt to guide the model. If None, a default prompt based on task_type will be used.
    :param task_type: The type of task, either "chart" or "geo".
    :return: The answer to the question.
    """
    # Initialize GPT4o client
    client = model_client.get_client()
    content = []
    if image_path is not None:
        if isinstance(image_path, str):
            image_path = [image_path]
        for image_path in image_path:
            base64_image = encode_image(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )
    content.append(
        {
            "type": "text",
            "text": question,
        }
    )
    # Prepare the message
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT if task_type == "chart" else GEO_SYSTEM_PROMPT
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    # Generate response
    response = client.generate(message)

    return response
