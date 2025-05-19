from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any
import logging

from mlx_parallm.server.schemas import (
    ModelList, InternalModelRecord, ModelCard,
    CompletionRequest, CompletionResponse, CompletionChoice, CompletionUsage
)
from mlx_parallm.utils import load as load_model_from_util
from mlx_parallm.utils import generate as generate_text_from_util

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
    Loads the initial model specified via CLI arguments using mlx_parallm.utils.load.
    """
    if current_server_args and current_server_args.model_path:
        model_id_cli = current_server_args.model_path # Use path as ID for now
        logging.info(f"Attempting to load initial model from CLI: {model_id_cli}")
        
        record = InternalModelRecord(
            id=model_id_cli,
            path_or_hf_id=current_server_args.model_path,
            status="loading",
            model_type="causal_lm" # Tentative type for models loaded via this path
        )
        model_registry[model_id_cli] = record

        try:
            # Actual model loading
            model_instance, tokenizer_instance = load_model_from_util(current_server_args.model_path)
            
            # Update the record in the registry
            record.model_instance = model_instance
            record.tokenizer_instance = tokenizer_instance
            record.status = "loaded"
            # model_type might be refined here if load_model_from_util provides more info
            logging.info(f"Successfully loaded model: {model_id_cli}")
        except Exception as e:
            record.status = "error_loading"
            logging.error(f"Failed to load model {model_id_cli}: {e}", exc_info=True)
        
        logging.info(f"Model {model_id_cli} registered with status: {record.status}")
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
async def list_models_endpoint():
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

@app.post("/v1/completions", response_model=CompletionResponse, tags=["Generation"])
async def create_completion(request: CompletionRequest):
    """
    Creates a completion for the provided prompt and parameters.
    """
    if request.model not in model_registry:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found in registry.")
    
    record = model_registry[request.model]
    if record.status != "loaded" or not record.model_instance or not record.tokenizer_instance:
        raise HTTPException(status_code=409, detail=f"Model '{request.model}' is not currently loaded or ready.")

    model = record.model_instance
    tokenizer = record.tokenizer_instance

    try:
        # Note: mlx_parallm.utils.generate expects temp and top_p in **kwargs
        # which are then passed to generate_step.
        # We need to ensure our CompletionRequest fields align or are transformed.
        # The current utils.generate takes: model, tokenizer, prompt, max_tokens, verbose, formatter, **kwargs
        # kwargs for generate_step includes: temp, top_p, repetition_penalty, logit_bias
        
        generation_kwargs = {
            "temp": request.temperature,
            "top_p": request.top_p
            # Add other supported params like repetition_penalty if added to CompletionRequest
        }

        generated_text = generate_text_from_util(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            # verbose=True, # Optional: for debugging
            **generation_kwargs
        )

        # Calculate token counts
        # Note: TokenizerWrapper might have a direct method, or we use the underlying tokenizer
        # This assumes tokenizer_instance is TokenizerWrapper and has an encode method for the prompt,
        # and can also encode the generated text.
        prompt_tokens = len(tokenizer.encode(request.prompt))
        completion_tokens = len(tokenizer.encode(generated_text))
        total_tokens = prompt_tokens + completion_tokens

        choice = CompletionChoice(
            text=generated_text,
            index=0,
            finish_reason="length" if completion_tokens >= request.max_tokens else "stop"
            # logprobs will be None for now
        )

        return CompletionResponse(
            model=request.model,
            choices=[choice],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )

    except Exception as e:
        logging.error(f"Error during text generation for model {request.model}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during text generation: {str(e)}")

# Further endpoints and application logic will be added here. 