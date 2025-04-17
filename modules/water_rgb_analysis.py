from typing import override

import models.perception
from modules.module import Module, ModuleResult


class WaterRGBAnalysis(Module):
    def __init__(self):
        super().__init__("WaterRGBAnalysis", {"WaterAnalysis", "RGBAnalysis", "SatelliteLoader"})

    @override
    def main(self) -> ModuleResult:
        # Benchmark: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation
        # model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",quantization_config=quantization_config, device_map="auto")

        model = models.perception.three_sixty_vl.ThreeSixtyVLModel()
        model.max_new_tokens = 8192

        water_analysis = self.load_from_file("water_analysis.txt")

        system_prompt: str = ""
        prompts: list[str] = [
            f"Given this description of the water bodies in the image: {water_analysis}, please describe how far any buildings are from the water and if there is a nature buffer zone between them.",
        ]

        location = self.load_location()

        model.image_paths = [f"./satellite_data/{location}/rgb.png"]
        output = '\n' + water_analysis + '\n' + model.multi_run_one_result(system_prompt, prompts)

        self.save_to_file("rgb_analysis.txt", output, append=True)
        return ModuleResult.OK
