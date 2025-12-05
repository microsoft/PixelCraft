import json
from typing import Optional, List, Dict, Any, Tuple
from src.prompts.prompt import Prompt


class PromptActionPlanner(Prompt):
    """Action planner for chart understanding tasks with step-by-step reasoning."""

    def __init__(self, tools_definition_path: str = "data/tools.json"):
        super().__init__()
        self.step_counter = 1
        self.current_image_path: Optional[str] = None
        self.current_output_path: Optional[str] = None
        self.history: List[str] = []
        self.question: Optional[str] = None
        self.image_pool: Optional[Dict] = None
        self.image_size: Optional[Any] = None
        self.tools_list: List[str] = []
        self.plan_reflection: Optional[str] = None
        self._load_tool_list(tools_definition_path)

    def _load_tool_list(self, definition_path: str) -> None:
        """Load tool definitions from JSON file."""
        with open(definition_path, "r") as f:
            self.tools = json.load(f)

    @staticmethod
    def _parse_tool_string(tool_str: str) -> List[str]:
        """Parse tool string '[tool1, tool2, tool3]' into list."""
        return tool_str.strip("[]").replace(" ", "").split(",")

    def get_user_prompt(self, *args, **kwargs) -> Tuple[str, str]:
        """Generate action planning prompt with current state and history."""
        self._update_parameters(**kwargs)

        plan_reflection_str = self._build_reflection_string()
        selected_tools = self._build_tools_description()
        base_prompt = self._build_base_prompt(selected_tools)

        history_str = "\n".join(self.history)

        full_prompt = self._assemble_full_prompt(
            base_prompt, plan_reflection_str, history_str
        )
        return full_prompt

    def _update_parameters(self, **kwargs) -> None:
        """Update internal parameters from kwargs."""
        param_map = {
            "image_path": "current_image_path",
            "output_path": "current_output_path",
            "question": "question",
            "original_task": "original_task",
            "step_counter": "step_counter",
            "history": "history",
            "image_pool": "image_pool",
        }

        for kwarg_key, attr_name in param_map.items():
            if kwarg_key in kwargs:
                setattr(self, attr_name, kwargs[kwarg_key])

        if "tool_selection_json" in kwargs and kwargs["tool_selection_json"]:
            self._process_tool_selection(kwargs["tool_selection_json"])

    def _process_tool_selection(self, tool_selection_json: Dict) -> None:
        """Process tool selection JSON to update tools list and reflection."""
        self.tool_selection_json = tool_selection_json

        if "tools" in tool_selection_json:
            tools = tool_selection_json["tools"]
            self.tools_list = (
                self._parse_tool_string(tools) if isinstance(tools, str) else tools
            )

        if "suggestion" in tool_selection_json:
            self.plan_reflection = tool_selection_json["suggestion"]

    def _build_reflection_string(self) -> str:
        """Build reflection string if available."""
        if not self.plan_reflection:
            return ""

        return (
            "The suggestion is provided for your reference:\n"
            f"{self.plan_reflection}\n"
            "You can follow the suggestion for tool selection and parameters setting."
        )

    def _build_tools_description(self) -> List[str]:
        """Build list of tool descriptions based on selection."""
        if self.tools_list:
            return [
                self._format_tool_description(self.tools[name])
                for name in self.tools_list
            ]
        return [self._format_tool_description(tool) for tool in self.tools.values()]

    def _format_tool_description(self, tool: Dict) -> str:
        """Format a single tool definition into readable string."""
        return json.dumps(tool, indent=2, ensure_ascii=False)

    def _build_base_prompt(self, selected_tools: List[str]) -> str:
        """Build the base prompt with instructions and tools."""
        return f"""
You are an expert on chart understanding to analyze the question step by step until answering the final answer.
You are given the chart figure <img src='{self.current_image_path}'> with image path being {self.current_image_path}, and need to solve the following question: {self.question}. 
The question can be solved via the intermediate processed image using tools.
To solve the complex task, you can decompose it to some simple subquestions and use tools to generate intermediate reasoning processes until getting the final answer.

Available Tools
Use the following tools to analyze and process images:

{",\n".join(selected_tools)}

Image pool:
{self.image_pool}

**Instructions**  
1. **Workflow**:  
   - Your actions should be logic and reasonable by preferentially considering the tools instead of outputting the final answer directly.
   - Each THOUGHT should analyze if the final answer is reached. If so, output the final answer with `TERMINATE` in the ACTION section.
   - You should first analyze the question and which tool can be used to answer the question according to the description of the tool.
   - You should consider the "Description" and "Constraints" of the tool, and make sure your tool call is reasonable and correct.
   - If the tool requires images, you should use the image path in the image pool, which includes the original image and the processed image from the previous steps.
   - If the tool execution fails, you should try to use other tools to answer the question.

2. **Final Answer**:
    - Considering all the reasoning processes, if you think you have reached the final answer, provide a concise summary of the solution.
    - Output the `FINAL ANSWER` in the `THOUGHT` section and 'TERMINATE' in the `ACTION` section to end the task.

3. **Format**:
    - Follow the provided format strictly for each step, like `THOUGHT N`, `ACTION N`, and `OBSERVATION N`.
    - Output analysis in the THOUGHT section and tool call in the ACTION section. Make sure THOUGHT and ACTION are in the each step.
    - When executing code, use the provided function signatures and return formats.
    - Use **absolute paths** for all files and the `output_path={self.current_output_path}`.  
    - Only output the function name and parameters in `ACTION` with clear kwargs.
    - **Do not include** the function description in `ACTION`.
    - End with `TERMINATE` in the ACTION section once the final answer is reached.
    - **Do not continue** after providing `FINAL ANSWER`.
    - Analyze which the tool can be used to answer the question first, and then generate the subquestion for the corresponding tool call if necessary.
    - If **no useful information** is found or **code error** happens in the last `OBSERVATION`, you should select **extract_information(image_path: str, question: str, related_info: str) -> dict** to answer the question in the next step.
"""

    def _assemble_full_prompt(
        self,
        base_prompt: str,
        plan_reflection_str: str,
        history_str: str,
    ) -> str:
        """Assemble the complete prompt with all components."""
        return f"""{base_prompt}

{plan_reflection_str}

Response History:
{history_str}

Current step template:
THOUGHT {self.step_counter}: [Your analysis for tool using].
ACTION {self.step_counter}: Your action code for tool using.
```python
tool_name(key=value)
```
OBSERVATION {self.step_counter}: [Result or observation].
"""

    def update_prompt(
        self,
        thought: str,
        question: Optional[str] = None,
        action_code: Optional[str] = None,
        observation: Optional[str] = None,
        new_image_path: Optional[str] = None,
        new_output_path: Optional[str] = None,
        image_pool: Optional[Dict] = None,
    ) -> str:
        """Update prompt with new step information and paths."""
        self._update_paths(new_image_path, new_output_path, question, image_pool)
        step_entry = self._build_step_entry(thought, action_code, observation)
        self.history.append("\n".join(step_entry))
        self.step_counter += 1
        return self.get_user_prompt()

    def _update_paths(
        self,
        new_image_path: Optional[str],
        new_output_path: Optional[str],
        question: Optional[str],
        image_pool: Optional[Dict],
    ) -> None:
        """Update current paths and parameters."""
        if new_image_path:
            self.current_image_path = new_image_path
        if new_output_path:
            self.current_output_path = new_output_path
        if question:
            self.question = question
        if image_pool:
            self.image_pool = image_pool

    def _build_step_entry(
        self, thought: str, action_code: Optional[str], observation: Optional[str]
    ) -> List[str]:
        """Build step entry with thought, action, and observation."""
        step_entry = [f"THOUGHT {self.step_counter}: {thought}"]

        if action_code:
            formatted_code = self._format_action_code(action_code)
            step_entry.append(f"ACTION {self.step_counter}:\n{formatted_code}")
        else:
            step_entry.append(f"ACTION {self.step_counter}: None")

        if observation:
            step_entry.append(f"OBSERVATION {self.step_counter}: {observation}")

        return step_entry

    def _format_action_code(self, action_code: str) -> str:
        """Format action code by replacing path placeholders."""
        return action_code.replace(
            "{image_path}", f'"{self.current_image_path}"'
        ).replace("{output_path}", f'"{self.current_output_path}"')


if __name__ == "__main__":
    prompt_subtask_planner = PromptActionPlanner()
    print(
        prompt_subtask_planner.get_messages(
            image_path="/path/to/image.png",
            question="question",
        )
    )
