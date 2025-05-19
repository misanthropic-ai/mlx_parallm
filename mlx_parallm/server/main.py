from fastapi import FastAPI, HTTPException, Depends
from starlette.responses import StreamingResponse
from typing import Dict, List, Any, AsyncGenerator, Tuple, Optional
import logging
import os
import time
import uuid
import mlx.core as mx
import mlx.nn as nn

from mlx_parallm.server.schemas import (
    ModelList, InternalModelRecord, ModelCard, ModelStatus,
    CompletionRequest, CompletionResponse, CompletionChoice, CompletionUsage,
    ChatCompletionRequest, ChatMessage, ChatCompletionChoice, ChatCompletionResponse,
    DeltaMessage, ChatCompletionStreamChoice, ChatCompletionChunk
)
from mlx_parallm.utils import (
    load as load_model_and_tokenizer_util,
    generate as generate_text_util,
    batch_stream_generate_text
)
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_parallm.cli import current_server_args # For accessing CLI args like model_path

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
            status=ModelStatus.LOADING,
            model_type="causal_lm"
        )
        model_registry[model_id_cli] = record

        try:
            # Actual model loading using the corrected alias
            model_instance, tokenizer_instance = load_model_and_tokenizer_util(current_server_args.model_path)
            
            # Update the record in the registry
            record.model_instance = model_instance
            record.tokenizer_instance = tokenizer_instance
            record.status = ModelStatus.LOADED
            # model_type might be refined here if load_model_from_util provides more info
            logging.info(f"Successfully loaded model: {model_id_cli}")
        except Exception as e:
            record.status = ModelStatus.ERROR_LOADING
            logging.error(f"Failed to load model {model_id_cli}: {e}", exc_info=True)
        
        logging.info(f"Model {model_id_cli} registered with status: {record.status.value}")
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
    if record.status != ModelStatus.LOADED or not record.model_instance or not record.tokenizer_instance:
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

        generated_text = generate_text_util(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
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

async def _chat_completion_stream_generator(
    request: ChatCompletionRequest,
    model_id: str,
    model_instance: nn.Module,
    tokenizer_instance: TokenizerWrapper
) -> AsyncGenerator[str, None]:
    """Generates text chunks for streaming chat completion in SSE format."""
    
    # 1. Prepare the prompt using the chat template
    try:
        messages_for_template = [msg.model_dump(exclude_none=True) for msg in request.messages]
        # For batch_stream_generate_text, we need a list of prompts, even if it's just one.
        # And it expects tokenized prompts.
        prompt_text = tokenizer_instance.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        logging.error(f"Error applying chat template for model {model_id} during stream: {e}")
        # In a stream, error handling like this is tricky. 
        # We might yield an error-formatted SSE or just stop.
        # For now, let it raise to be caught by FastAPI error handling if possible before stream starts,
        # or simply log and stop if error occurs mid-stream prep.
        # A robust solution would involve yielding an error SSE.
        # Simplified: if template fails, we won't even start the generate_step stream.
        # This specific error would occur before yielding anything.
        raise HTTPException(status_code=500, detail=f"Error processing chat messages for streaming: {str(e)}")

    # Tokenize the prompt(s). batch_stream_generate_text expects tokenized input.
    # Ensure left-padding if it becomes relevant for future batch > 1 streaming.
    # For now, with batch_size = 1, padding strategy is less critical.
    if tokenizer_instance._tokenizer.pad_token is None:
        tokenizer_instance._tokenizer.pad_token = tokenizer_instance.eos_token
        # tokenizer_instance._tokenizer.pad_token_id = tokenizer_instance.eos_token_id # Already handled by TokenizerWrapper typically
    tokenizer_instance._tokenizer.padding_side = 'left' # Good practice for generation

    # We are processing one request at a time from the API perspective for now.
    # So, a batch of one prompt_text.
    # `tokenizer_instance._tokenizer` is the underlying Hugging Face tokenizer.
    prompts_tokens_list = [prompt_text] # List of one prompt string
    try:
        # We need mx.array(tokenizer._tokenizer(prompts_list, padding=True)['input_ids'])
        # For a single prompt, padding=False is fine, but batch_stream_generate expects a batch.
        encoded_prompts = tokenizer_instance._tokenizer(prompts_tokens_list, return_tensors="np", padding=False) # Use np for mx.array
        prompts_mx_array = mx.array(encoded_prompts["input_ids"])
    except Exception as e:
        logging.error(f"Error tokenizing prompt for model {model_id} during stream: {e}")
        raise HTTPException(status_code=500, detail=f"Error tokenizing prompt for streaming: {str(e)}")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:28]}"
    first_chunk = True

    generation_kwargs = {
        "temp": request.temperature if request.temperature is not None else 0.7,
        "top_p": request.top_p if request.top_p is not None else 1.0,
        # Add other relevant parameters from request if batch_stream_generate_text supports them
    }

    try:
        # Assuming batch_stream_generate_text handles a batch of 1 correctly.
        # It yields List[Tuple[Optional[str], Optional[str]]]
        for batch_deltas_and_reasons in batch_stream_generate_text(
            model_instance,
            tokenizer_instance, # Already a TokenizerWrapper
            prompts_mx_array,
            request.max_tokens or 1024,
            **generation_kwargs
        ):
            # For now, we are handling n=1, so we only care about the first element of the batch.
            text_delta, finish_reason = batch_deltas_and_reasons[0]

            choice_delta = DeltaMessage()
            if first_chunk and text_delta is not None: # text_delta can be empty string initially
                choice_delta.role = "assistant"
                first_chunk = False
            
            if text_delta is not None:
                choice_delta.content = text_delta
            
            if finish_reason:
                # If there's a final text_delta along with finish_reason, ensure it's included.
                # This should be handled by batch_stream_generate_text logic.
                pass # delta content is already set if any

            # Only send a chunk if there's content or it's a finish chunk
            if choice_delta.content or choice_delta.role or finish_reason:
                stream_choice = ChatCompletionStreamChoice(
                    index=0, 
                    delta=choice_delta, 
                    finish_reason=finish_reason
                )
                chunk = ChatCompletionChunk(
                    id=request_id, 
                    model=model_id, 
                    choices=[stream_choice]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            if finish_reason:
                break # Stop streaming for this request if finished

    except Exception as e:
        logging.error(f"Error during model generation stream for {model_id}: {e}", exc_info=True)
        # Yield a final error message in SSE format if possible, or just log.
        # For simplicity, we'll just stop. A robust implementation might send an error SSE.
        error_delta = DeltaMessage(content=f"\n\nError during generation: {str(e)}")
        error_choice = ChatCompletionStreamChoice(index=0, delta=error_delta, finish_reason="error")
        error_chunk = ChatCompletionChunk(id=request_id, model=model_id, choices=[error_choice])
        yield f"data: {error_chunk.model_dump_json()}\n\n" # Try to send an error chunk
    finally:
        yield f"data: [DONE]\n\n"

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    model_id = request.model
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found or not loaded.")

    internal_record = model_registry[model_id]
    if internal_record.status != ModelStatus.LOADED or internal_record.model_instance is None or internal_record.tokenizer_instance is None:
        raise HTTPException(status_code=500, detail=f"Model '{model_id}' is not ready for chat. Status: {internal_record.status.value}")

    model_instance = internal_record.model_instance
    # Ensure tokenizer_instance is a TokenizerWrapper for batch_stream_generate_text and apply_chat_template
    if not isinstance(internal_record.tokenizer_instance, TokenizerWrapper):
        tokenizer_instance = TokenizerWrapper(internal_record.tokenizer_instance)
    else:
        tokenizer_instance = internal_record.tokenizer_instance

    if request.stream:
        logging.info(f"Streaming request received for model {model_id}. Stream flag: {request.stream}")
        # TODO: Handle n > 1 for streaming if supported by batch_stream_generate_text properly.
        # For now, n=1 is assumed by how stream_generator processes batch_deltas_and_reasons[0]
        if request.n is not None and request.n > 1:
            raise HTTPException(status_code=400, detail="Streaming with n > 1 is not currently supported.")
        return StreamingResponse(
            _chat_completion_stream_generator(request, model_id, model_instance, tokenizer_instance),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming (existing logic)
        try:
            messages_for_template = [msg.model_dump(exclude_none=True) for msg in request.messages]
            prompt_text = tokenizer_instance.apply_chat_template(
                messages_for_template,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logging.error(f"Error applying chat template for model {model_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing chat messages: {str(e)}")

        try:
            # Note: generate_text_util is for single, non-batched, non-streaming generation.
            # If we make generate_text_util batch-aware, this could be simplified.
            # For now, it works for n=1.
            generation_kwargs = {
                "temp": request.temperature if request.temperature is not None else 0.7,
                "top_p": request.top_p if request.top_p is not None else 1.0,
                # repetition_penalty: request.repetition_penalty, # if supported and added to schema
            }
            generated_text = generate_text_util(
                model_instance,
                tokenizer_instance,
                prompt_text,
                max_tokens=request.max_tokens or 1024,
                **generation_kwargs
            )
        except Exception as e:
            logging.error(f"Error during text generation for model {model_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during text generation: {str(e)}")

        prompt_tokens = len(tokenizer_instance.encode(prompt_text))
        completion_tokens = len(tokenizer_instance.encode(generated_text))
        total_tokens = prompt_tokens + completion_tokens

        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )

        assistant_message = ChatMessage(role="assistant", content=generated_text.strip())
        # TODO: Handle n > 1 for non-streaming if generate_text_util is made to support it.
        choice = ChatCompletionChoice(index=0, message=assistant_message, finish_reason="stop") 

        return ChatCompletionResponse(
            model=model_id,
            choices=[choice],
            usage=usage
        )

# Further endpoints and application logic will be added here. 