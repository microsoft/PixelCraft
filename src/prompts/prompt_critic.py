from src.prompts.prompt import Prompt

class PromptVisualCritic(Prompt):
    def __init__(self):
        super().__init__()

    def get_user_prompt(self, **kwargs):
        return kwargs.get("critique_prompt", "")


class PromptPlanCritic(Prompt):
    def __init__(self):
        super().__init__()

    def get_user_prompt(self, **kwargs):
        return kwargs.get("plan_critique_prompt", "")
