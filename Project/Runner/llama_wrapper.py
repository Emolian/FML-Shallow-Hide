import os
import contextlib
import yaml
from llama_cpp import Llama


class LlamaWrapper:
    def __init__(self, config_path="Config/model_setting.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Absolutize the model path
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_rel_path = self.config["model"]["path"]
        model_path = os.path.join(base_dir, model_rel_path)

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self.max_tokens = self.config["model"].get("max_tokens", 256)

        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stderr(fnull):
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    verbose=False
                )

    def _load_config(self, path):
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), "..", path)
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def generate(self, prompt: str) -> str:
        response = self.llm(prompt, max_tokens=self.max_tokens)
        return response["choices"][0]["text"].strip()

