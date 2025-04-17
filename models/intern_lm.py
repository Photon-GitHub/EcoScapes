from transformers import AutoTokenizer, AutoModelForCausalLM

from models.model import Model


class InternLM(Model):
    model = None
    tokenizer = None

    def __init__(self) -> None:
        super().__init__("internlm/internlm2_5-7b-chat")

        if InternLM.model is None or InternLM.tokenizer is None:
            InternLM.model = AutoModelForCausalLM.from_pretrained(
                self.name,
                quantization_config=self.quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            InternLM.tokenizer = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)

    def run(self, system_prompt: str, prompt: str) -> str:
        response, _ = InternLM.model.chat(self.tokenizer, system_prompt + " " + prompt, history=[], max_new_tokens=self.max_new_tokens)
        return response
