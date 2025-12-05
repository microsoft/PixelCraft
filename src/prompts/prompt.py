from abc import ABC
from typing import Optional, List
from src.utils.tool_utils import encode_image


class Prompt(ABC):
    def __init__(self, system_prompt: str = None, user_prompt: str = ""):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def get_system_prompt(self) -> str:
        """get system prompt"""
        return self.system_prompt

    def get_user_prompt(self, **kwargs) -> str:
        """get user prompt"""
        return self.user_prompt

    def get_messages(self, images: Optional[List[str]] = None, **kwargs) -> list:
        """get complete messages list format for model input"""
        messages = []

        system_prompt = self.get_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_prompt = self.get_user_prompt(**kwargs)

        content = []
        if images:
            if not isinstance(images, list):
                images = [images]
            for image in images:
                if not image.startswith("data:image"):
                    image = encode_image(image)
                content.append({"type": "image_url", "image_url": {"url": image}})
            content.append({"type": "text", "text": user_prompt})

        else:
            content = user_prompt

        messages.append({"role": "user", "content": content})

        return messages
