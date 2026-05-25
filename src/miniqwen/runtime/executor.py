from miniqwen.config import Qwen3Config
from miniqwen.model.text_model import Qwen3TextModel


class RuntimeExecutor(Qwen3TextModel):
    def __init__(self, config: Qwen3Config, state_dict: dict, device=None, compute_dtype=None):
        super().__init__(config, state_dict, device=device, compute_dtype=compute_dtype)
