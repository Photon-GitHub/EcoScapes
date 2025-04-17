from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from models.perception.perception_model import PerceptionModel


# Gives only very abstract answers.
class LLavaNextModel(PerceptionModel):
    processor = None
    model = None

    def __init__(self):
        super().__init__("llava-hf/llava-v1.6-mistral-7b-hf")

        if LLavaNextModel.model is None or LLavaNextModel.processor is None:
            LLavaNextModel.processor = LlavaNextProcessor.from_pretrained(self.name)
            LLavaNextModel.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.name,
                quantization_config=self.quantization_config,
                device_map='auto',
                low_cpu_mem_usage=True,
            )

        self.processor = LLavaNextModel.processor
        self.model = LLavaNextModel.model

    def run(self, system_prompt: str, prompt: str) -> str:
        processor = self.processor
        model = self.model

        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt + " " + prompt},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=self.load_images(), text=prompt, return_tensors="pt")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        return processor.decode(output[0], skip_special_tokens=True)
