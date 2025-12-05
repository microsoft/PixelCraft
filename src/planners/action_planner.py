import os
import re
import copy

from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image
from loguru import logger

from autogen.oai.client import OpenAIWrapper
from autogen.agentchat import Agent, ConversableAgent
from autogen.agentchat.contrib.img_utils import (
    gpt4v_formatter,
    message_formatter_pil_to_b64,
)
from autogen.code_utils import content_str
from autogen._pydantic import model_dump

from src.utils.model import ModelClient
from src.prompts.manager import prompt_manager
from src.agents.tool_agents import ImageTools
from src.planners.summarizer import Summarizer
from src.agents.reasoner import Reasoner


DEFAULT_LMM_SYS_MSG = """You are a helpful AI assistant."""


class ActionPlanner(ConversableAgent):
    def __init__(
        self,
        name: str,
        task_type: str = "chart",
        system_message: Optional[Union[str, List]] = DEFAULT_LMM_SYS_MSG,
        is_termination_msg: str = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            name (str): agent name.
            system_message (str): system message for the OpenAIWrapper inference.
                Please override this attribute if you want to reprogram the agent.
            **kwargs (dict): Please refer to other kwargs in
                [ConversableAgent](../conversable_agent#__init__).
        """
        super().__init__(
            name,
            system_message,
            is_termination_msg=is_termination_msg,
            *args,
            **kwargs,
        )

        self.task_type = task_type
        self.tool_agent = ImageTools(task_type=task_type)
        self.client = ModelClient()
        self.stop = None

        self.prompt_action_planner = prompt_manager.get_prompt("action_planner")
        self.reasoner = Reasoner()
        self.summarizer = Summarizer()

        # Override the `generate_oai_reply`, to register the reply function.
        self.replace_reply_func(
            ConversableAgent.generate_oai_reply, ActionPlanner.generate_oai_reply
        )
        self.replace_reply_func(
            ConversableAgent.a_generate_oai_reply,
            ActionPlanner.a_generate_oai_reply,
        )

    def update_system_message(self, system_message: Union[Dict, List, str]):
        """Update the system message.

        Args:
            system_message (str): system message for the OpenAIWrapper inference.
        """
        self._oai_system_message[0]["content"] = self._message_to_dict(system_message)[
            "content"
        ]
        self._oai_system_message[0]["role"] = "system"

    @staticmethod
    def _message_to_dict(message: Union[Dict, List, str]) -> Dict:
        """Convert a message to a dictionary. This implementation
        handles the GPT-4V formatting for easier prompts.

        The message can be a string, a dictionary, or a list of dictionaries:
            - If it's a string, it will be cast into a list and placed in the 'content' field.
            - If it's a list, it will be directly placed in the 'content' field.
            - If it's a dictionary, it is already in message dict format. The 'content' field of this dictionary
            will be processed using the gpt4v_formatter.
        """
        if isinstance(message, str):
            return {"content": gpt4v_formatter(message, img_format="pil")}
        if isinstance(message, list):
            return {"content": message}
        if isinstance(message, dict):
            assert "content" in message, "The message dict must have a `content` field"
            if isinstance(message["content"], str):
                message = copy.deepcopy(message)
                message["content"] = gpt4v_formatter(
                    message["content"], img_format="pil"
                )
            try:
                content_str(message["content"])
            except (TypeError, ValueError) as e:
                print(
                    "The `content` field should be compatible with the content_str function!"
                )
                raise e
            return message
        raise ValueError(f"Unsupported message type: {type(message)}")

    def generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""
        if messages is None:
            messages = self._oai_messages[sender]

        messages_with_b64_img = message_formatter_pil_to_b64(
            self._oai_system_message + messages
        )

        print('messages[-1].pop("context", None): ', messages[-1].pop("context", None))
        if messages[-1].pop("context", None) is not None:
            logger.warning("Context in the last message is not None, ignoring it.")
            return False, None

        extracted_response = self.client.generate(messages_with_b64_img, stop=self.stop)

        if not isinstance(extracted_response, str):
            extracted_response = model_dump(extracted_response)
        return True, extracted_response

    def direct_answer(
        self, question: str, image_path: str = None, task_type: str = "chart"
    ) -> str:
        if task_type == "chart":
            question += (
                "\n\nAnalyze the chart step by step and then answer the question."
            )
        return self.reasoner.reason(subquery=question, image_path=image_path)

    def reset(self):
        """Reset the agent. Do not reset the client since 'GPT4o' object has no attribute 'clear_usage_summary'."""
        self.clear_history()
        self.reset_consecutive_auto_reply_counter()
        self.stop_reply_at_receive()
        self.prompt_action_planner = prompt_manager.get_prompt("action_planner")
        # if self.client is not None:
        #     self.client.clear_usage_summary()
        for reply_func_tuple in self._reply_func_list:
            if reply_func_tuple["reset_config"] is not None:
                reply_func_tuple["reset_config"](reply_func_tuple["config"])
            else:
                reply_func_tuple["config"] = copy.copy(reply_func_tuple["init_config"])

    def react(
        self,
        question: str,
        image_path: str = None,
        output_dir: str = None,
        max_steps: int = 5,
        tool_selection_json: str = None,
        original_task: str = None,
        direct_answer: bool = False,
        error_tolerance: int = 1,
    ) -> str:
        """core pipeline of the action planner"""
        image_pool = {"original_image": image_path, "processed_image": []}
        self.original_image_path = image_path
        self.question = question
        self.instruction = question.replace(original_task, "")
        output_path = f"{output_dir}/1.jpg"
        image_original = Image.open(image_path)
        image_size = image_original.size
        self.prompt_action_planner.image_size = image_size
        prompt = self.prompt_action_planner.get_user_prompt(
            question=question,
            image_path=image_path,
            output_path=output_path,
            image_pool=image_pool,
            tool_selection_json=tool_selection_json,
            original_task=original_task,
        )

        self.image_path_description = {}
        self.tool_call_history = []
        messages = []

        for step in range(1, max_steps + 1):
            if direct_answer:
                break
            logger.info(f"Step {step}...")
            stop = [f"OBSERVATION {step}"]

            logger.debug(f"Prompt: {prompt}")
            messages = self.client.build_messages(prompt=prompt, image_path=image_path)
            response = self.client.generate(
                messages=messages,
                stop=stop,
            )
            logger.debug(f"Response: {response}")

            # parse the response
            try:
                thought, action = self._parse_response(response, step)
                logger.info(f"Thought: {thought}")

                # check termination condition
                if self._check_termination(action):
                    final_answer = ""
                    if len(self.instruction) > 0 and (
                        "grounded to a number that is exlicitly written"
                        in self.instruction
                    ):
                        logger.info(f"Pre_Answer: {thought}")
                        final_answer = self.summarizer.summarize(
                            image_path=image_path,
                            original_task=original_task,
                            final_answer=thought,
                            instructions=question.replace(original_task, ""),
                        )

                    logger.info(f"Final Answer: {thought}\n{final_answer}")
                    self._append_oai_message(
                        response, "assistant", self, is_sending=True
                    )
                    return (
                        self._format_final_answer(thought + "\n" + final_answer),
                        self._update_final_message(
                            messages, thought + "\n" + final_answer, action, step
                        ),
                        self.tool_call_history,
                    )

                if (
                    "extract_information" in action
                    and self.original_image_path in action
                ):
                    observation = ""
                    direct_answer = True
                else:
                    (
                        observation,
                        current_image_path,
                        model_response,
                        self.tool_call_history,
                        self.image_path_description,
                        executed_successfully,
                    ) = self.tool_agent.execute_action(
                        action,
                        self.question,
                        self.instruction,
                        self.tool_call_history,
                        self.image_path_description,
                    )
                    if (
                        current_image_path != self.original_image_path
                        and current_image_path is not None
                        and os.path.exists(current_image_path)
                    ):
                        image_pool["processed_image"].append(current_image_path)
                    if model_response is not None:
                        inter_reason = model_response
                    else:
                        inter_reason = ""

                    observation += inter_reason
                    logger.info(f"Observation: {observation}")
                    if os.path.exists(output_path):
                        output_path = f"{output_dir}/{step + 1}.jpg"

                    if not executed_successfully:
                        error_tolerance -= 1
                        if error_tolerance <= 0:
                            direct_answer = True
                            logger.info("Tolerance exceeded, switching to direct answer.")

            except Exception as e:
                thought, action = None, None
                logger.error(f"Error parsing response: {str(e)}")
                observation = (
                    f"Code Error: {str(e)}, please use other tools for reasoning."
                )

            prompt = self.prompt_action_planner.update_prompt(
                thought=thought,
                action_code=action,
                observation=observation,
                new_output_path=output_path,
                image_pool=image_pool,
            )

        # after max steps or direct answer
        logger.info("Reached maximum steps or direct answer required, generating final answer...")
        response = self.direct_answer(question, image_path, task_type=self.task_type)
        final_answer = ""
        if len(self.instruction) > 0 and (
            "grounded to a number that is exlicitly written" in self.instruction
        ):
            final_answer = self.summarizer.summarize(
                image_path=image_path,
                original_task=original_task,
                final_answer=response,
                instructions=question.replace(original_task, ""),
            )
        logger.info(f"Final Answer: {response}\n{final_answer}")
        return (
            response + "\n" + final_answer,
            self._update_final_message(
                messages, response + "\n" + final_answer, action="TERMINATE", step=step
            ),
            self.tool_call_history,
        )

    def _parse_response(self, response: str, step: int) -> tuple:
        """parse the response to get thought and action"""
        try:
            pattern = re.compile(
                r"THOUGHT\s*{}:\s*(.*?)\s*ACTION\s*{}:\s*(.*)".format(step, step),
                re.DOTALL,
            )
            match = pattern.search(response)

            if not match:
                if "TERMINATE" in response:
                    return response.replace("TERMINATE", "").strip(), "TERMINATE"
                raise ValueError("Invalid response format")

            return match.group(1).strip(), match.group(2).strip()
        except Exception as e:
            raise RuntimeError(f"Response parsing failed: {str(e)}")

    def _check_termination(self, action: str) -> bool:
        """check if the action is TERMINATE"""
        return "TERMINATE" in action.upper()

    def _format_final_answer(self, thought: str) -> str:
        """format the final answer"""
        return f"✅ Final Answer: {thought.strip()}"

    def _update_context(
        self, context: list, step: int, thought: str, action: str, observation: str
    ) -> list:
        """update the context with the new thought, action and observation"""
        new_context = copy.deepcopy(context)

        new_context.extend(
            [
                {"role": "assistant", "content": f"THOUGHT {step}: {thought}"},
                {"role": "assistant", "content": f"ACTION {step}: {action}"},
                {"role": "system", "content": f"OBSERVATION {step}: {observation}"},
            ]
        )
        return new_context

    def _update_final_message(self, message, thought: str, action: str, step: int):
        """update the final message with the final thought and action"""
        if not message:
            return message
        content = message[-1].get("content")
        if content and isinstance(content, list) and len(content) > 1 and isinstance(content[1], dict):
            content[1]["text"] += (
                f"\n\nTHOUGHT {step}: {thought}\nACTION {step}: {action}"
            )
        else:
            logger.warning(f"Unexpected message content format: {type(content)}")
        return message

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """Receive a message from another agent.

        Once a message is received, this function will generate REACT by calling tools to answer the question step by step until TERMINATE.
        prompt
        llm: generate THOUGHT and ACTION
        parser: parse the action and generate execution plan with python code
        executor: execute the python code and generate the result
        """
        self._process_received_message(message, sender, silent)

        if self._is_termination_msg(message):
            return

        if (
            request_reply is False
            or request_reply is None
            and self.reply_at_receive[sender] is False
        ):
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)

    def initiate_chat(self, assistant, message, log_prompt_only=False):
        if log_prompt_only:
            print(message)
        else:
            assistant.receive(message, self, request_reply=True)
