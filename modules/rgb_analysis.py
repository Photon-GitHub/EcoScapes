from typing import override

import models.perception
from modules.module import Module, ModuleResult


class RGBAnalysis(Module):
    def __init__(self):
        super().__init__("RGBAnalysis", {"SatelliteLoader"})

    @override
    def main(self) -> ModuleResult:
        # Benchmark: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation
        # model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",quantization_config=quantization_config, device_map="auto")

        model = models.perception.three_sixty_vl.ThreeSixtyVLModel()
        model.max_new_tokens = 8192
        system_prompt: str = ""  # ("You are an expert satellite image analyst. "
        # "Assume North is up, South is down, West is left, and East is right.")

        prompts: list[str] = [
            "Please approximate the size and population of the town or city.",
            "Is the town or city layout grid-like, circular, or irregular?",
            "Where in the picture is the city center located? Please describe the location and the distance from the center to the edges of the town.",
            "Are there any forests or parks in the image? If yes, please describe their location and size.",
            "Are there any railway lines or stations in the image? If yes, please describe their location and connections.",
            "Are there any major bridges or tunnels in the image? If yes, please describe their location and connections.",
            "How are the buildings distributed in the town? Are there any residential, commercial, industrial or mixed zones in the image? Please describe their location and density, if it is visible from the picture. Otherwise, say that you are unable to identify specific zones.",
            "What is the approximate distribution and density of buildings in the town or city? Are there clusters of high-density areas?",
            "Can you identify large open spaces within the urban area? Describe their locations and approximate sizes.",
            "Are there distinct patterns in the road network? Describe the layout and any major intersections.",
            "Is there a visible distinction between urban and rural areas? Describe the transition zones and their locations.",
            "Can you identify any major landmarks or large structures that stand out? Describe their locations relative to the city center.",
            "Are there large expanses of greenery or parkland visible? Describe their locations and approximate sizes.",
            "Can you see any significant geographical features, such as hills or valleys, in or around the city? Describe their locations.",
        ]

        location = self.load_location()

        model.image_paths = [f"./satellite_data/{location}/rgb.png"]
        output = model.multi_run_one_result(system_prompt, prompts)

        self.save_to_file("rgb_analysis.txt", output)
        return ModuleResult.OK
