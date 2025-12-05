from src.utils.model import ModelClient
from src.prompts.manager import prompt_manager


class Reasoner:
    def __init__(self):
        self.client = ModelClient()
        self.prompt_reasoner = prompt_manager.get_prompt("reasoner")

    def reason(self, subquery: str, image_path: str = None) -> str:
        """
        Main reasoning workflow
        :param subquery: Target subquery identifier
        :param image_path: Path to input image
        :return: Generated response string
        """
        messages = self.prompt_reasoner.get_messages(
            images=image_path,
            subquery=subquery
        )
        response = self.client.generate(messages)
        return response


if __name__ == "__main__":
    reasoner = Reasoner()
    response = reasoner.reason(
        subquery="What is the main object in the image?",
        image_path="data/CharXiv/images/0.jpg"
    )
    print(response)
