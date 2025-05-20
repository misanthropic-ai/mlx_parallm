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
    batch_stream_generate_text,
    stream_generate as stream_generate_text_util
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

async def _completion_stream_generator(
    request: CompletionRequest,
    model_id: str,
    model_instance: nn.Module,
    tokenizer_instance: TokenizerWrapper
) -> AsyncGenerator[str, None]:
    """Generates text chunks for streaming completions in SSE format."""
    
    request_id = f"cmpl-{uuid.uuid4().hex[:29]}"

    generation_kwargs = {
        "temp": request.temperature,
        "top_p": request.top_p,
        # Add other relevant parameters from CompletionRequest if stream_generate_text_util supports them
    }

    try:
        # stream_generate_text_util yields text segments directly
        for text_delta in stream_generate_text_util(
            model_instance,
            tokenizer_instance,
            request.prompt,
            max_tokens=request.max_tokens,
            **generation_kwargs
        ):
            if text_delta is not None: # stream_generate can yield empty strings or None, ensure we send valid data
                choice = CompletionChoice(
                    text=text_delta,
                    index=0,
                    finish_reason=None # Finish reason is sent in the last chunk
                )
                # For completions, each stream event is a full CompletionResponse-like object
                chunk = CompletionResponse(
                    id=request_id,
                    object="text_completion", # OpenAI uses "text_completion" for stream chunks too
                    created=int(time.time()), # New timestamp for each chunk
                    model=model_id,
                    choices=[choice],
                    usage=None # Usage is typically not sent with each chunk, but with the final non-streamed response
                               # Or as a separate extension if supported. For OpenAI compat, it's not in chunks.
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        
        # After the loop, send a final chunk with finish_reason
        # The stream_generate function doesn't explicitly return a finish_reason itself.
        # We have to infer it. If the loop completed without breaking due to EOS (which stream_generate handles internally),
        # and we reached here, it means max_tokens might have been hit or generation naturally concluded.
        # For simplicity, let's assume "stop" if loop finishes, or "length" if max_tokens was the constraint.
        # This part needs refinement based on how stream_generate signals completion type.
        # Assuming stream_generate stops at EOS or max_tokens.
        # A more robust way would be to get the actual finish reason from the generation process.
        # For now, we send a final chunk with a presumptive finish_reason. Let's assume "stop" for now.
        # The last text_delta might be empty if EOS was the last token.

        # OpenAI typically sends the last chunk with the finish_reason set.
        # It does not send content in this very last message usually if reason is stop/length.
        final_choice = CompletionChoice(
            text="", # Usually empty text in the final chunk with finish_reason
            index=0,
            finish_reason="stop" # Default to stop, could be length if max_tokens reached
                                 # stream_generate should ideally provide this or we count tokens.
        )
        final_chunk = CompletionResponse(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=model_id,
            choices=[final_choice],
            usage=None
        )
        yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

    except Exception as e:
        logging.error(f"Error during model generation stream for {request.model}: {e}", exc_info=True)
        # In a real scenario, you might want to send an error formatted SSE message here
        # For now, this will break the stream, and the client might see a connection drop or incomplete stream.
        # Example of error SSE (not fully implemented here):
        # error_payload = {"error": {"message": str(e), "type": "internal_error", "code": None}}
        # yield f"data: {json.dumps(error_payload)}\n\n"
        pass # Let the stream break for now
    finally:
        yield f"data: [DONE]\n\n"

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

    model_instance = record.model_instance
    # Ensure tokenizer_instance is TokenizerWrapper for stream_generate_text_util
    if not isinstance(record.tokenizer_instance, TokenizerWrapper):
        tokenizer_instance = TokenizerWrapper(record.tokenizer_instance)
    else:
        tokenizer_instance = record.tokenizer_instance

    # Add log to inspect request.stream before the conditional block
    logging.info(f"Inside create_completion for model {request.model}. Parsed stream flag: {request.stream}, type: {type(request.stream)}")

    if request.stream:
        # TODO: Handle n > 1 for streaming if stream_generate_text_util is adapted for it.
        if request.n is not None and request.n > 1:
            raise HTTPException(status_code=400, detail="Streaming with n > 1 is not currently supported for completions.")
        logging.info(f"Streaming completion request received for model {request.model}. Stream flag: {request.stream}")
        return StreamingResponse(
            _completion_stream_generator(request, request.model, model_instance, tokenizer_instance),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming logic (remains largely the same)
        try:
            generation_kwargs = {
                "temp": request.temperature,
                "top_p": request.top_p
            }

            generated_text = generate_text_util(
                model=model_instance,
                tokenizer=tokenizer_instance,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                **generation_kwargs
            )

            prompt_tokens = len(tokenizer_instance.encode(request.prompt))
            completion_tokens = len(tokenizer_instance.encode(generated_text))
            total_tokens = prompt_tokens + completion_tokens

            # For non-streaming, finish_reason needs to be determined based on completion_tokens vs max_tokens
            finish_reason_val = "length" if completion_tokens >= request.max_tokens else "stop"

            choice = CompletionChoice(
                text=generated_text,
                index=0,
                finish_reason=finish_reason_val 
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