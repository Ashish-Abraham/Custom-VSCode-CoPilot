# filename: model_server.py
from kserve import Model, ModelServer
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from typing import List, Dict
from huggingface_hub._login import _login

_login(token='your_token_here', add_to_git_credential=False)

class StarCoder(Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.ready = False
       self.tokenizer = None
       #
       self.model_id = 'bigcode/starcoderbase'
       self.load()

    def load(self):
        # this step fetches the model from huggingface directly. the downloads may take longer and be slow depending on upstream link. We recommend using TIR Models
        # instead
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                          trust_remote_code=True,
                                                          device_map='auto')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.pipeline  = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        inputs = payload["instances"]
        source_text = inputs[0]["text"]
        # Encode the source text
        inputs = self.tokenizer.encode(source_text, return_tensors="pt").to(self.device)
        # Generate sequences
        sequences = self.model.generate(inputs,
                                        do_sample=True,
                                        top_k=10,
                                        num_return_sequences=1,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        max_length=200)
        results = []
        for seq in sequences:
            results.append(self.tokenizer.decode(seq))
        return {"predictions": results}

if __name__ == "__main__":
    model = StarCoder("starcoderbase")
    ModelServer().start([model])


