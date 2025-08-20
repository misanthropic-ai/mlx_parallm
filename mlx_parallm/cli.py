import uvicorn
from pydantic import Field
from pydantic_cli import run_and_exit, Cmd
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold server arguments for access by the FastAPI app
# This is a simple approach for now; can be replaced by a more robust config system.
current_server_args: Optional["ServerCLIArgs"] = None

class ServerCLIArgs(Cmd):
    # model_config = pydantic.ConfigDict(cli_prog_name="mlx_parallm_serve") # Optional: for help message

    model_path: str = Field(..., description="The path or Hugging Face ID of the base model to load.", cli=["--model-path"])
    host: str = Field("127.0.0.1", description="Host to bind the server to.", cli=["--host"])
    port: int = Field(8000, description="Port to bind the server to.", cli=["--port"])
    lora_path: Optional[str] = Field(
        None,
        description="Optional path to a LoRA/DoRA adapter to load at startup.",
        cli=["--lora-path"],
    )
    max_batch_size: int = Field(8, description="Max batch size for dynamic batching.", cli=["--max-batch-size"])
    batch_timeout: float = Field(0.1, description="Batching window in seconds.", cli=["--batch-timeout"])
    request_timeout_seconds: float = Field(600.0, description="Per-request processing timeout (seconds).", cli=["--request-timeout-seconds"])
    max_concurrent_streams: int = Field(4, description="Limit for concurrent streaming responses to protect batch throughput.", cli=["--max-concurrent-streams"])
    # We'll add more arguments like workers, log_level, config_file later
    # Log level for Uvicorn can be set directly in uvicorn.run

    def run(self):
        """
        Starts the Uvicorn server with the specified arguments.
        This method is called by pydantic-cli.
        """
        global current_server_args
        current_server_args = self # Store the instance of ServerCLIArgs

        logger.info(f"Starting server with initial model: {self.model_path}")
        logger.info(f"Server will listen on {self.host}:{self.port}")

        # The actual model loading based on self.model_path will happen
        # within the FastAPI app's startup sequence, where it can access
        # current_server_args and populate the model registry.

        uvicorn.run(
            "mlx_parallm.server.main:app", # Path to the FastAPI app instance
            host=self.host,
            port=self.port,
            log_level="info",  # Default Uvicorn log level
            reload=False # Set to True for development if you want auto-reload
            # workers=self.workers # Add when 'workers' arg is implemented
        )

# The entry point function for the CLI script
def cli_runner():
    run_and_exit(ServerCLIArgs, description="MLX ParaLLM Server CLI", version="0.1.0")

if __name__ == "__main__":
    # This allows running the CLI directly for testing, e.g., `python mlx_parallm/cli.py --model-path my/model`
    cli_runner() 
