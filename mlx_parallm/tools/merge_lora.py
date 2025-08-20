from pydantic import Field
from pydantic_cli import Cmd, run_and_exit


class MergeLoRAArgs(Cmd):
    base_model: str = Field(..., description="Base model path or HF ID.", cli=["--base-model"])
    lora_path: str = Field(..., description="Path to LoRA/DoRA adapter weights.", cli=["--lora-path"])
    output_path: str = Field(..., description="Directory to write merged model.", cli=["--output-path"])
    format: str = Field("mlx", description="Output format: 'mlx' or 'safetensors' (placeholder)", cli=["--format"])

    def run(self):
        # Placeholder stub; full merge is non-trivial and will be implemented later.
        print("merge_lora stub — not implemented yet.")
        print(f"Would merge {self.lora_path} into {self.base_model} and write to {self.output_path} in format={self.format}")


def merge_lora_cli_runner():
    run_and_exit(MergeLoRAArgs, description="Merge LoRA adapter into base model (stub)", version="0.1.0")

