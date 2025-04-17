import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from models.model import Model


# TODO: Needs Triton to run, which is not available for windows.
class PhiThreeSmall(Model):
    def __init__(self) -> None:
        super().__init__("microsoft/Phi-3-small-128k-instruct")

    def run(self, system_prompt: str, prompt: str) -> str:
        model = AutoModelForCausalLM.from_pretrained(
            self.name,
            quantization_config=self.quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        assert torch.cuda.is_available(), "This model needs a GPU to run ..."
        device = torch.cuda.current_device()
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        generation_args = {
            "max_new_tokens": self.max_new_tokens,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args)
        return output[0]['generated_text']
