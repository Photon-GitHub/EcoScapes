import os.path
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Set


class ModuleResult(Enum):
    OK = auto()
    STOP_PIPELINE = auto()
    ERROR = auto()


class Module(ABC):
    def __init__(self, name: str, dependencies: Set[str] = None, soft_dependencies: Set[str] = None):
        """
        Initialize the module with a name, its dependencies, and optional soft dependencies.

        :param name: Name of the module.
        :param dependencies: Set of dependencies for the module.
        :param soft_dependencies: Set of optional soft dependencies for the module.
        """
        self.name = name
        self.dependencies = dependencies if dependencies is not None else set()
        self.soft_dependencies = soft_dependencies if soft_dependencies is not None else set()

    @abstractmethod
    def main(self) -> ModuleResult:
        """
        The main method that should be implemented by subclasses.

        :return: A ModuleResult indicating the outcome of the operation.
        """
        pass

    def load_location(self, file_path="./module_communication/location_name.txt") -> str:
        """
        Load location name from a file.

        :param file_path: Path to the file containing the location name.
        :return: Location name read from the file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    def load_from_file(self, file_name: str) -> str:
        """
        Load the content of a file from the module communication directory.

        :param file_name: Name of the file to be read.
        :return: Content of the file.
        """
        # Define the directory path
        location = self.load_location()
        dir_path = os.path.realpath(f"./module_communication/{location}/")

        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)

        with open(os.path.join(dir_path, file_name), "r") as file:
            return file.read()

    def save_to_file(self, file_name: str, text: str, append: bool = False):
        """
        Save text to a file in the module communication directory.

        :param file_name: Name of the file to be written to.
        :param text: Text content to be written to the file.
        :param append: Whether to append to the file (default is False).
        """
        # Define the directory path
        location = self.load_location()
        dir_path = os.path.realpath(f"./module_communication/{location}/")

        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)

        # Write the text to the file
        with open(os.path.join(dir_path, file_name), "a" if append else "w") as file:
            file.write(text)
