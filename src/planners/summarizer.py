from src.utils.model import ModelClient
from src.prompts.prompt_summarizer_ground import PromptSummarizer


class Summarizer:
    def __init__(self):
        # Initialize predefined subtask prompt templates
        self.prompts = {
            "describe_image": "Describe the main content in the image {image_path}.",
            "analyze_scene": "Analyze the scene type and primary elements in image {image_path}.",
        }
        self.client = ModelClient()
        self.prompt_summarizer = PromptSummarizer()

    def summarize(
        self,
        image_path: str,
        original_task: str,
        final_answer: str,
        stop=None,
        model_response=None,
        instructions=None,
    ) -> str:
        """
        Main reasoning workflow
        :param subtask: Target subtask identifier
        :param image_path: Path to input image
        :return: Generated response string
        """
        prompt = self.prompt_summarizer.get_user_prompt(
            original_task=original_task,
            final_answer=final_answer,
            model_response=model_response,
            instructions=instructions,
        )

        messages = self.client.build_messages(
            prompt=prompt,
            image_path=image_path,
        )

        response = self.client.generate(
            messages=messages,
            stop=stop,
        )

        return response

if __name__ == "__main__":
    summarizer = Summarizer()
    result = summarizer.summarize(
        image_path="data/CharXiv/images/0.jpg",
        original_task="Summarize the content of the image.",
        final_answer="The image shows a beautiful sunset over the mountains.",
    )
    print(result)
