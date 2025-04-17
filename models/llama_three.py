from transformers import AutoTokenizer, AutoModelForCausalLM

from models.model import Model


# TODO: Does not work due to licensing
class LlamaThree(Model):
    def __init__(self) -> None:
        super().__init__("meta-llama/Meta-Llama-3-8B-Instruct")

    def run(self, system_prompt: str, prompt: str) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        model = AutoModelForCausalLM.from_pretrained(
            self.name,
            quantization_config=self.quantization_config,
            device_map="auto",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = tokenizer.decode(response, skip_special_tokens=True)
        return decoded_response.strip()
