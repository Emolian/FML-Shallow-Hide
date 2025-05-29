import os
import contextlib
import yaml
from llama_cpp import Llama


class LlamaWrapper:
    def __init__(self, config_path="Config/model_setting.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Resolve base model path
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_rel_path = self.config["model"]["path"]
        model_path = os.path.join(base_dir, model_rel_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Model generation settings
        self.max_tokens = self.config["model"].get("max_tokens", 256)
        self.temperature = self.config["model"].get("temperature", 0.3)  # Lower temp = more focused
        self.top_k = self.config["model"].get("top_k", 40)
        self.top_p = self.config["model"].get("top_p", 0.95)
        self.stop_tokens = self.config["model"].get("stop", ["###"])  # Default: stop at next prompt block

        # Load model silently
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
        response = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            stop=self.stop_tokens
        )
        return response["choices"][0]["text"].strip()

