from src.prompts.prompt import Prompt

class PromptCritic(Prompt):
    def __init__(self):
        super().__init__()

    def get_prompt(self, *args, **kwargs) -> str:
        description = kwargs['description']
        
        # 构建基础提示
        prompt = f'''You are a data analysis expert specializing in visual chart interpretation. Given a chart image and the description. Your task is to critique whether the description is strictly aligned with the chart.

[Description]
{description}

[Output Guidelines]
1. Strictly evaluate the description:
- Verify if the description accurately reflects the chart information.
- Check for misinterpretations (e.g., mislabeled axes, incorrect value extractions).
2. If partially correct, regard it as incorrect.
3. Output format:
[Analysis]: <Your analysis of the description>
[Critique]: True/False

'''
        
        return prompt


if __name__ == '__main__':
    prompt_subtask_planner = PromptCritic()
    print(prompt_subtask_planner.get_prompt(image_path_code="image_path_code", question="question", sorted_bbox_json="sorted_bbox_json"))