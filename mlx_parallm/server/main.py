from fastapi import FastAPI
from typing import Dict, List
import logging

from mlx_parallm.server.schemas import ModelList, InternalModelRecord, ModelCard

# Attempt to import server arguments from cli.py
# This is a simple way to pass CLI args; will be refined with a proper config system.
try:
    from mlx_parallm.cli import current_server_args
except ImportError:
    current_server_args = None # type: ignore
    logging.warning("Could not import current_server_args from mlx_parallm.cli. Server might not have model path if not run via CLI.")

app = FastAPI(
    title="mlx_parallm Server",
    version="0.1.0",
    description="A high-performance, parallelized batch generation server for MLX models.",
)

# Simple in-memory model registry
# Key: model_id (str), Value: InternalModelRecord
model_registry: Dict[str, InternalModelRecord] = {}

@app.on_event("startup")
async def startup_event():
    """
    Event handler for application startup.
    Currently used to load the initial model specified via CLI arguments.
    """
    if current_server_args and current_server_args.model_path:
        model_id = current_server_args.model_path # Use path as ID for now
        # In a real scenario, you would attempt to load the model here.
        # For now, we just register it as 'loaded'.
        # The actual loading logic will be complex and involve mlx_lm.load().
        logging.info(f"Registering initial model from CLI: {model_id}")
        record = InternalModelRecord(
            id=model_id,
            path_or_hf_id=current_server_args.model_path,
            status="loaded", # Assume loaded for now; actual loading logic will update this
            model_type="causal_lm" # Placeholder, should be determined during actual load
        )
        model_registry[model_id] = record
        logging.info(f"Model {model_id} registered with status: {record.status}")
    else:
        logging.warning("No initial model path found in server arguments. Model registry will be empty at startup.")

@app.get("/health", tags=["General"])
async def health_check():
    """
    Endpoint to check the health of the server.
    Returns a simple status indicating the server is operational.
    """
    return {"status": "ok"}

@app.get("/v1/models", response_model=ModelList, tags=["Models"])
async def list_models():
    """
    Lists all models available to the server, along with their current status.
    Follows OpenAI's API format for /v1/models.
    """
    model_cards: List[ModelCard] = []
    for record in model_registry.values():
        model_cards.append(record.to_model_card())
    
    # TODO: Add logic to list "available_not_loaded" models
    # This would involve checking a configuration (e.g., a list of discoverable models
    # from a config file or a directory scan) and adding them to the list if not already in the registry.
    # For now, it only lists models explicitly loaded (or simulated as loaded).

    return ModelList(data=model_cards)

# Further endpoints and application logic will be added here. 