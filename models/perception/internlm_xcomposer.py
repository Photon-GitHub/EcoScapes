import torch
from transformers import AutoModel, AutoTokenizer

from models.perception.perception_model import PerceptionModel


class InternLMXComposerModel(PerceptionModel):
    def __init__(self):
        # super().__init__('internlm/internlm-xcomposer2-4khd-7b')
        super().__init__("internlm/internlm-xcomposer2-vl-1_8b")

    def run(self, system_prompt: str, prompt: str) -> str:
        torch.set_grad_enabled(False)

        # init model and tokenizer
        model = AutoModel.from_pretrained(self.name, trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)

        query = '<ImageHere>' + system_prompt + " " + prompt
        image = self.image_paths[0]

        with torch.cuda.amp.autocast():
            response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)

        return response
