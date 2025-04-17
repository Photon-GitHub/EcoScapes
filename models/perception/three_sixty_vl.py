import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.perception.perception_model import PerceptionModel


class ThreeSixtyVLModel(PerceptionModel):
    model = None
    tokenizer = None

    def __init__(self) -> None:
        super().__init__("qihoo360/360VL-8B")

        if ThreeSixtyVLModel.model is None or ThreeSixtyVLModel.tokenizer is None:
            ThreeSixtyVLModel.model = AutoModelForCausalLM.from_pretrained(self.name,
                                                                           quantization_config=self.quantization_config,
                                                                           device_map='auto',
                                                                           low_cpu_mem_usage=True,
                                                                           trust_remote_code=True).eval()
            ThreeSixtyVLModel.tokenizer = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
            ThreeSixtyVLModel.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = ThreeSixtyVLModel.model
        self.tokenizer = ThreeSixtyVLModel.tokenizer

        self.terminators = [self.tokenizer.convert_tokens_to_ids("<|eot_id|>", )]

        self.vision_tower = self.model.get_vision_tower()
        self.vision_tower.load_model()
        self.vision_tower.to(device="cuda", dtype=torch.float16)
        self.image_processor = self.vision_tower.image_processor

    def run(self, system_prompt: str, prompt: str) -> str:
        image = self.load_images()[0].convert("RGB")

        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=f"{system_prompt} {prompt}", image=image, image_processor=self.image_processor)

        input_ids = inputs["input_ids"].to(device='cuda', non_blocking=True)
        images = inputs["image"].to(dtype=torch.float16, device='cuda', non_blocking=True)

        output_ids = self.model.generate(
            input_ids,
            images=images,
            do_sample=False,
            eos_token_id=self.terminators,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            temperature=0.7,
            use_cache=True)

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        return outputs
