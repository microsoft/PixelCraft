import os
import time
from loguru import logger
from openai import OpenAI
from src.utils.tool_utils import encode_image


def get_custom_model(base_url="http://localhost:8000/v1"):
    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    return client

class ModelClient:
    def __init__(self, base_url=None, model_name="gpt-4.1-mini-2025-04-14"):
        self.model_name = os.environ.get(
            "DEPLOYMENT_NAME", model_name
        )
        base_url = base_url or os.environ.get("BASE_URL")
        if base_url:
            self.client = get_custom_model(base_url=base_url)
        else:
            logger.warning("Using OpenAI's official API endpoint.")
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )

    def get_client(self):
        return self.client

    def build_messages(
        self,
        prompt,
        image_path=None,
        system_prompt=None,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        content = []
        if image_path:
            encoded_image = encode_image(image_path)
            content.append({"type": "image_url", "image_url": {"url": encoded_image}})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def generate(
        self,
        messages,
        max_retries=3,
        max_tokens=10000,
        temperature=0.0,
        seed=42,
        top_p=None,
        top_k=None,
        stop=None,
        repetition_penalty=None,
    ):
        llm_config = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        if top_p is not None:
            llm_config["top_p"] = top_p
        if stop is not None:
            llm_config["stop"] = stop
        if top_k is not None or repetition_penalty is not None:
            llm_config["extra_body"] = {}
            if top_k is not None:
                llm_config["extra_body"]["top_k"] = top_k
            if repetition_penalty is not None:
                llm_config["extra_body"]["repetition_penalty"] = repetition_penalty
        for retry in range(max_retries):
            try:
                # response = await asyncio.to_thread(
                #     self.client.chat.completions.create,
                #     model=self.model_name,
                #     messages=messages,
                #     **llm_config,
                # )
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **llm_config,
                )
                extracted_response = response.choices[0].message.content
                return extracted_response
            except Exception as e:
                if retry == max_retries - 1:
                    return f"All retries failed: {str(e)}"
                # await asyncio.sleep(1)
                time.sleep(1)

