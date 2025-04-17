from typing import override

import models.perception
from modules.module import Module, ModuleResult


class WaterAnalysis(Module):
    def __init__(self):
        super().__init__("WaterAnalysis", {"WaterPreprocessing", "SatelliteLoader"})

    @override
    def main(self) -> ModuleResult:
        # Benchmark: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation
        # model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",quantization_config=quantization_config, device_map="auto")
        model = models.perception.three_sixty_vl.ThreeSixtyVLModel()
        model.max_new_tokens = 8192
        system_prompt: str = "The map shows water as white and land as black. A river is a very long, connected, white area. A lake is a large, circular, white area."
        prompts: list[str] = [
            "Is the map depicting a lake? If it does not, please say so.",
            "Is the map depicting a river? If it does not, please say so.",
            "Is the map depicting a part of the coast? If it does not, please say so.",
        ]

        location = self.load_location()

        model.image_paths = [f"./satellite_image_processing/{location}/water_preprocessed.png"]
        output = [prompt + " - " + model.run(system_prompt, prompt) for prompt in prompts]
        output = "\n".join(output)

        self.save_to_file("water_analysis.txt", output)
        return ModuleResult.OK
