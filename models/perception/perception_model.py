import os
from abc import ABC
from typing import List

import torch
from PIL import Image
from transformers import BitsAndBytesConfig

from models.model import Model


class PerceptionModel(Model, ABC):
    """
    A subclass of Model that includes perception capabilities with input images.

    Attributes:
        name (str): The name of the model, defined at initialization.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the PerceptionModel with a name.

        Args:
            name (str): The name of the model.
        """
        super().__init__(name)
        self._image_paths: List[str] = []

    @property
    def image_paths(self) -> List[str]:
        """
        Gets the image paths for the model.

        Returns:
            List[str]: A list of image paths.
        """
        return self._image_paths

    @image_paths.setter
    def image_paths(self, value: List[str]) -> None:
        """
        Sets the image paths for the model, converting relative paths to absolute paths.

        Args:
            value (List[str]): A list of image paths to set.
        """
        self._image_paths = [os.path.abspath(path) for path in value]

    def load_images(self) -> List[Image.Image]:
        """
        Lazily loads the images from the paths.

        Returns:
            List[Image.Image]: A list of loaded images.
        """
        return [Image.open(path) for path in self._image_paths]
