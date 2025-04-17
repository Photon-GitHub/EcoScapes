from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from models.perception.perception_model import PerceptionModel


# TODO: NOT FUNCTIONAL, THROWS ERRORS
class ChatTruthModel(PerceptionModel):

    def __init__(self):
        super().__init__("mingdali/ChatTruth-7B")

    def run(self, system_prompt: str, prompt: str) -> str:
        # use cuda device
        model = AutoModelForCausalLM.from_pretrained(self.name, device_map='auto', trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)

        model.generation_config = GenerationConfig.from_pretrained(self.name, trust_remote_code=True)
        model.generation_config.top_p = 0.01
        model.generation_config.max_new_tokens = self.max_new_tokens

        query = tokenizer.from_list_format([
            {'image': self.image_paths[0]},
            {'text': system_prompt + " " + prompt},
        ])

        response, _ = model.chat(tokenizer, query=query, history=None)
        return response
