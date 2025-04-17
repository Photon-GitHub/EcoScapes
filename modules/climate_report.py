from typing import override

import models.llama_three
from modules.module import Module, ModuleResult


class ClimateReport(Module):
    def __init__(self):
        super().__init__("ClimateReport", dependencies={"RGBAnalysis", "MoistureAnalysis"}, soft_dependencies={"WaterRGBAnalysis"})

    @override
    def main(self) -> ModuleResult:
        # Benchmark: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation
        # model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",quantization_config=quantization_config, device_map="auto")

        model = models.intern_lm.InternLM()
        model.max_new_tokens = 16384

        location = self.load_location()
        original_prompt = (
            f"You will be given a report of the RGB satellite image of the city or town {location} and its surrounding area and a description of moisture anomalies of the town and its surroundings taken on a sunny day."
            " Please use this information to write a report on the current state of climate adaptation of the town."
            " Only focus on the current situation and do not make any predictions.")

        rgb_analysis = self.load_from_file("rgb_analysis.txt")
        moisture_analysis = self.load_from_file("moisture_analysis.txt")

        system_prompt = ("You are a climate scientist with a focus on climate adaptation."
                         " You will be given tasks that will result in a report to analyse the current state of climate adaptation in a city or town."
                         " Answer accurately, informatively and in a neutral way that aligns with the scientific consensus.")

        prompt = original_prompt + " The RGB satellite image description: " + rgb_analysis + " The moisture map description: " + moisture_analysis
        output = model.run(system_prompt, prompt)

        self.save_to_file("climate_report.txt", output)
        return ModuleResult.OK
