from typing import override

import models.perception
from modules.module import Module, ModuleResult


class MoistureAnalysis(Module):
    def __init__(self):
        super().__init__("MoistureAnalysis", {"SatelliteLoader"})

    @override
    def main(self) -> ModuleResult:
        # Benchmark: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation
        # model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",quantization_config=quantization_config, device_map="auto")

        model = models.perception.three_sixty_vl.ThreeSixtyVLModel()
        model.max_new_tokens = 8192

        system_prompt: str = ""

        prompts: list[str] = [
            "List specific spots with the highest heat levels (red areas).",
            "List specific spots with the lowest heat levels (blue areas).",
            "Compare the heat levels between different sectors (e.g., urban vs. rural).",
            "Identify patterns or trends in heat distribution (e.g., is there a gradient?).",
            "Analyze potential reasons for red high heat spots (e.g., industrial areas, lack of vegetation).",
            "Analyze potential reasons for blue low heat spots (e.g., water bodies, dense vegetation).",
            "Assess the implications of heat distribution on urban infrastructure.",
        ]

        location = self.load_location()
        model.image_paths = [f"./satellite_data/{location}/moisture.png"]
        output = model.multi_run_one_result(system_prompt, prompts)

        self.save_to_file("moisture_analysis.txt", output)
        return ModuleResult.OK
