from typing import Dict, Type, Optional, Any
from .prompt import Prompt
from .prompt_action_planner import PromptActionPlanner
from .prompt_reasoner import PromptReasoner
from .prompt_critic import PromptVisualCritic, PromptPlanCritic


class PromptManager:
    """Manages prompt classes and their instantiation with configurable defaults."""
    
    def __init__(self, default_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize PromptManager with optional default configurations.
        
        Args:
            default_configs: Optional dict mapping prompt names to their default init parameters.
                           Example: {"action_planner": {"tools_definition_path": "data/tools.json"}}
        """
        self._prompts: Dict[str, Type[Prompt]] = {
            "action_planner": PromptActionPlanner,
            "reasoner": PromptReasoner,
            "visual_critic": PromptVisualCritic,
            "plan_critic": PromptPlanCritic,
        }
        self._default_configs = default_configs or {}

    def register_prompt(self, name: str, prompt_class: Type[Prompt], default_config: Optional[Dict[str, Any]] = None):
        """
        Register a new prompt class with optional default configuration.
        
        Args:
            name: Prompt name identifier
            prompt_class: Prompt class to register
            default_config: Optional default initialization parameters
        """
        self._prompts[name] = prompt_class
        if default_config:
            self._default_configs[name] = default_config

    def get_prompt(self, name: str, **kwargs) -> Prompt:
        """
        Get prompt instance by name with optional parameters.
        
        Args:
            name: Prompt name identifier
            **kwargs: Override parameters for prompt initialization
            
        Returns:
            Initialized Prompt instance
            
        Raises:
            ValueError: If prompt name is not registered
        """
        if name not in self._prompts:
            raise ValueError(f"Unknown prompt: {name}. Available prompts: {self.list_prompts()}")

        # Merge default config with provided kwargs (kwargs take precedence)
        init_params = self._default_configs.get(name, {}).copy()
        init_params.update(kwargs)
        
        # Instantiate with merged parameters or no parameters if none provided
        return self._prompts[name](**init_params) if init_params else self._prompts[name]()

    def list_prompts(self) -> list:
        """List all registered prompt names."""
        return list(self._prompts.keys())
    
    def update_default_config(self, name: str, config: Dict[str, Any]) -> None:
        """
        Update default configuration for a specific prompt.
        
        Args:
            name: Prompt name identifier
            config: New default configuration parameters
        """
        if name not in self._prompts:
            raise ValueError(f"Unknown prompt: {name}")
        self._default_configs[name] = config


# Global instance with default configurations
prompt_manager = PromptManager(
    default_configs={
        "action_planner": {"tools_definition_path": "data/tools.json"},
    }
)
