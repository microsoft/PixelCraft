from src.prompts.prompt import Prompt

class PromptReasoner(Prompt):
    def __init__(self):
        super().__init__()

    def get_system_prompt(self):
        return "You are an expert in chart analysis and data visualization. Please analyze the chart carefully and provide accurate answers based on the visual data presented."

    def get_user_prompt(self, **kwargs):
        return kwargs.get("subquery", "")