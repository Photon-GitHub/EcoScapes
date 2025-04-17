import os
from typing import override

from modules.module import Module, ModuleResult


class LocationExtraction(Module):
    def __init__(self):
        super().__init__("LocationExtraction")

    @override
    def main(self) -> ModuleResult:
        location = "Rosstal"

        dir_path = os.path.realpath("./module_communication/")

        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)

        # Write the text to the file
        with open(os.path.join(dir_path, "location_name.txt"), "w") as file:
            file.write(location)

        return ModuleResult.OK
