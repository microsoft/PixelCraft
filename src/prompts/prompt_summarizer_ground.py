from src.prompts.prompt import Prompt


class PromptSummarizer(Prompt):
    def __init__(self):
        super().__init__()

    def get_user_prompt(self, *args, **kwargs) -> str:
        original_task = kwargs["original_task"]
        final_answer = kwargs["final_answer"]

        prompt = f"""You are given an image of one or more scientific charts,
a natural-language question about the chart(s): {original_task}
and the answer to the question: {final_answer}
 
Your task is to ground the answer to a number that is exlicitly written in the chart, even if it's an approximate value.
If the answer is already grounded to a number that is exlicitly written in the chart, you can return the answer directly.
Do not change the answer, only ground it if needed.

Output format:
<Analysis if the answer is grounded to a number that is exlicitly written in the chart>
<Select the closest number that is exlicitly written in the chart>
"""

        return prompt
