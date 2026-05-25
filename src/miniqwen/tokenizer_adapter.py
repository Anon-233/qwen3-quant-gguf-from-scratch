class TokenizerAdapter:
    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

    def encode(self, text: str):
        return self.tokenizer(text, return_tensors="pt").input_ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
