from abc import ABC, abstractmethod

import torch
from transformers import BitsAndBytesConfig


class Model(ABC):
    """
    Abstract base class to model a Large Language Model (LLM).

    Attributes:
        name (str): The name of the model, defined at initialization.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the Model with a name.

        Args:
            name (str): The name of the model.
        """
        self.name: str = name
        self._max_new_tokens: int = 4096
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    @property
    def max_new_tokens(self) -> int:
        """
        Gets the maximum number of new tokens the model can generate.

        Returns:
            int: The maximum number of new tokens.
        """
        return self._max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value: int) -> None:
        """
        Sets the maximum number of new tokens the model can generate.

        Args:
            value (int): The maximum number of new tokens to set.
        """
        self._max_new_tokens = value

    @abstractmethod
    def run(self, system_prompt: str, prompt: str) -> str:
        """
        Abstract method to run the model. This method should be implemented by subclasses.

        Args:
            system_prompt (str): The system prompt for the model. If the model does not require or support a system prompt, this will be prepended to the prompt.
            prompt (str): The user prompt for the model. Always required.

        Returns:
            str: The result of running the model.
        """
        pass

    def multi_run_one_result(self, system_prompt: str, prompts: list[str]) -> str:
        """
        This runs the model with multiple prompts and returns the concatenated results.

        Args:
            system_prompt (str): The system prompt for the model. If the model does not require or support a system prompt, this will be prepended to the prompt.
            prompts (list[str]): The list of user prompts for the model. Always required.

        Returns:
            str: The result of running the model.
        """
        return "\n".join([self.run(system_prompt, prompt) for prompt in prompts])
