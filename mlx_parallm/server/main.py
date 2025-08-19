from fastapi import FastAPI, HTTPException, Depends
from starlette.responses import StreamingResponse
from typing import Dict, List, Any, AsyncGenerator, Tuple, Optional, Union
import logging
import os
import time
import uuid
import mlx.core as mx
import mlx.nn as nn
import asyncio
from asyncio import Future
import json
from collections import defaultdict

from mlx_parallm.server.schemas import (
    ModelList, InternalModelRecord, ModelCard, ModelStatus,
    CompletionRequest, CompletionResponse, CompletionChoice, CompletionUsage,
    ChatCompletionRequest, ChatMessage, ChatCompletionChoice, ChatCompletionResponse,
    DeltaMessage, ChatCompletionStreamChoice, ChatCompletionChunk,
    PerplexityRequest, PerplexityResponse
)
from mlx_parallm.utils import (
    load as load_model_and_tokenizer_util,
    generate as generate_text_util,
    batch_stream_generate_text,
    stream_generate as stream_generate_text_util,
    batch_generate_text as batch_generate_text_util
)
from mlx_parallm.sample_utils import top_p_sampling
import numpy as np
import math
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

# Request Queue and Batching Configuration
REQUEST_QUEUE: asyncio.Queue = asyncio.Queue()
MAX_BATCH_SIZE = 8  # Default value, can be made configurable later
BATCH_TIMEOUT = 0.1 # Seconds, default value, can be made configurable later
REQUEST_TIMEOUT_SECONDS = 60.0 # Default request timeout

class QueuedRequest:
    future: Future
    request_data: Union[CompletionRequest, ChatCompletionRequest]
    # request_id: str # Optional: for better tracking/logging

    def __init__(self, request_data: Union[CompletionRequest, ChatCompletionRequest]):
        self.future = Future()
        self.request_data = request_data
        # self.request_id = f"req_{uuid.uuid4().hex}"

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
            # Actual model loading
            model_instance, tokenizer_instance = load_model_and_tokenizer_util(
                current_server_args.model_path
            )
            
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

    # Start the batch processing worker
    asyncio.create_task(batch_processing_worker())
    logging.info("Batch processing worker task created.")

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

    # If token-level outputs are requested, handle synchronously (non-batched)
    if (request.logprobs is not None and request.logprobs > 0) or (request.echo is True):
        logging.info("Handling completion with logprobs/echo synchronously.")
        return await _compute_completion_with_logprobs(
            model_instance,
            tokenizer_instance,
            request.model,
            request
        )

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
        # Non-streaming logic
        # The batching worker now handles 'n', so direct generation here assumes n=1 or is overridden by queue.
        # This path is for requests not going through the queue if queue is bypassed or for single immediate processing.
        # For consistency with batching, 'n' support for non-streaming is primarily via the batch worker.
        # If a request with n > 1 reaches here directly (e.g. if queueing is disabled), it might not produce n results.
        # However, all non-streaming requests are intended to go through the batching worker.
        logging.info(f"Non-streaming completion request for model {request.model}. It will be queued for batch processing.")
        
        queued_req = QueuedRequest(request_data=request)
        await REQUEST_QUEUE.put(queued_req)
        
        try:
            # Wait for the result from the batch processing worker
            # Add a timeout to prevent indefinite waiting
            response_data = await asyncio.wait_for(queued_req.future, timeout=REQUEST_TIMEOUT_SECONDS)
            return response_data
        except asyncio.TimeoutError:
            logging.error(f"Request timed out for model {request.model} after {REQUEST_TIMEOUT_SECONDS} seconds.")
            raise HTTPException(status_code=504, detail="Request processing timed out.")
        except Exception as e:
            # Handle other exceptions that might be set on the future by the worker
            logging.error(f"Error processing request from queue for model {request.model}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def _compute_completion_with_logprobs(
    model_instance: nn.Module,
    tokenizer_instance: TokenizerWrapper,
    model_id: str,
    request: CompletionRequest,
):
    # Tokenize prompt
    enc = tokenizer_instance._tokenizer([request.prompt], return_tensors="np", padding=False)
    prompt_ids = enc["input_ids"]
    prompt_mx = mx.array(prompt_ids)
    B, L = prompt_mx.shape
    assert B == 1

    max_tokens = request.max_tokens
    temperature = request.temperature
    top_p = request.top_p
    topk = int(request.logprobs) if request.logprobs else 0

    # Echo: compute prompt token logprobs (teacher-forcing)
    echo_tokens = []
    echo_token_logprobs = []
    echo_top_logprobs = []
    if request.echo:
        logits_full = model_instance(prompt_mx)  # (1, L, V)
        toks = prompt_ids[0].tolist()
        for i in range(L - 1):
            logits_i = logits_full[:, i, :]
            # Apply logit_bias if provided
            if request.logit_bias:
                for k, v in request.logit_bias.items():
                    try:
                        tid = int(k)
                    except ValueError:
                        tid = tokenizer_instance._tokenizer.convert_tokens_to_ids(k)
                    if tid is not None and tid >= 0:
                        logits_i[:, tid] = logits_i[:, tid] + float(v)
            if temperature and temperature > 0:
                logits_i = logits_i * (1.0 / temperature)
            probs_i = mx.softmax(logits_i, axis=-1)
            token_id = int(prompt_mx[0, i + 1].item())
            lp = float(mx.log(probs_i[0, token_id]).item())
            echo_tokens.append(tokenizer_instance._tokenizer.convert_ids_to_tokens([toks[i + 1]])[0])
            echo_token_logprobs.append(lp)
            if topk > 0:
                pi = np.array(probs_i[0])
                idx = pi.argsort()[::-1][:topk]
                echo_top_logprobs.append({
                    tokenizer_instance._tokenizer.convert_ids_to_tokens([int(j)])[0]: float(np.log(pi[j]))
                    for j in idx
                })

    # Generate with per-step logprobs
    y = prompt_mx
    generated_ids = []
    gen_token_logprobs = []
    gen_top_logprobs = []
    for _ in range(max_tokens):
        logits = model_instance(y)
        logits_last = logits[:, -1, :]
        # Apply logit_bias if provided
        if request.logit_bias:
            for k, v in request.logit_bias.items():
                try:
                    tid = int(k)
                except ValueError:
                    tid = tokenizer_instance._tokenizer.convert_tokens_to_ids(k)
                if tid is not None and tid >= 0:
                    logits_last[:, tid] = logits_last[:, tid] + float(v)
        logits_sample = logits_last * (1.0 / temperature) if (temperature and temperature > 0) else logits_last
        probs = mx.softmax(logits_sample, axis=-1)
        if top_p is not None and 0.0 < top_p < 1.0:
            token = top_p_sampling(logits_sample, top_p, temperature if temperature > 0 else 1.0)
        else:
            token = mx.argmax(logits_sample, axis=-1, keepdims=True)
        tok_id = int(token[0, 0].item())
        generated_ids.append(tok_id)
        gen_token_logprobs.append(float(mx.log(probs[0, tok_id]).item()))
        if topk > 0:
            pi = np.array(probs[0])
            idx = pi.argsort()[::-1][:topk]
            gen_top_logprobs.append({
                tokenizer_instance._tokenizer.convert_ids_to_tokens([int(j)])[0]: float(np.log(pi[j]))
                for j in idx
            })
        y = mx.concatenate([y, token], axis=1)
        if tok_id == tokenizer_instance.eos_token_id:
            break

    # Assemble text
    gen_text = tokenizer_instance._tokenizer.decode(generated_ids)
    full_text = (request.prompt + gen_text) if request.echo else gen_text

    # Build logprobs object if requested
    logprobs_obj = None
    if topk > 0:
        if request.echo:
            tokens = echo_tokens + tokenizer_instance._tokenizer.convert_ids_to_tokens(generated_ids)
            token_logprobs = echo_token_logprobs + gen_token_logprobs
            top_logprobs = echo_top_logprobs + gen_top_logprobs
        else:
            tokens = tokenizer_instance._tokenizer.convert_ids_to_tokens(generated_ids)
            token_logprobs = gen_token_logprobs
            top_logprobs = gen_top_logprobs
        logprobs_obj = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "text_offset": [0] * len(tokens),
        }

    usage = CompletionUsage(
        prompt_tokens=int(prompt_mx.shape[1]),
        completion_tokens=len(generated_ids),
        total_tokens=int(prompt_mx.shape[1]) + len(generated_ids),
    )

    choice = CompletionChoice(text=full_text, index=0, logprobs=logprobs_obj, finish_reason="stop")
    return CompletionResponse(model=model_id, choices=[choice], usage=usage)

@app.post("/v1/perplexity", response_model=PerplexityResponse, tags=["Analysis"])
async def compute_perplexity(request: PerplexityRequest):
    model_id = request.model
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found or not loaded.")
    rec = model_registry[model_id]
    if rec.status != ModelStatus.LOADED or rec.model_instance is None or rec.tokenizer_instance is None:
        raise HTTPException(status_code=409, detail=f"Model '{model_id}' is not ready for perplexity.")

    model_instance = rec.model_instance
    tok = rec.tokenizer_instance
    if not isinstance(tok, TokenizerWrapper):
        tok = TokenizerWrapper(tok)

    enc = tok._tokenizer([request.text], return_tensors="np", padding=False)
    ids = enc["input_ids"][0].tolist()
    if len(ids) < 2:
        return PerplexityResponse(model=model_id, token_count=0, avg_nll=0.0, ppl=1.0)
    x = mx.array([ids])
    logits = model_instance(x)  # (1, T, V)
    T = logits.shape[1]
    logprobs = []
    for i in range(T - 1):
        li = logits[:, i, :]  # (1, V)
        pi = mx.softmax(li, axis=-1)
        tgt = int(x[0, i + 1].item())
        lp = float(mx.log(pi[0, tgt]).item())
        logprobs.append(lp)
    if not logprobs:
        return PerplexityResponse(model=model_id, token_count=0, avg_nll=0.0, ppl=1.0)
    avg_nll = -sum(logprobs) / len(logprobs)
    ppl = math.exp(avg_nll)
    return PerplexityResponse(model=model_id, token_count=len(logprobs), avg_nll=avg_nll, ppl=ppl)

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
        # Ensure non-streaming requests with n > 1 are handled by the batch worker
        logging.info(f"Non-streaming chat completion request for model {model_id}. It will be queued for batch processing.")
        
        queued_req = QueuedRequest(request_data=request)
        await REQUEST_QUEUE.put(queued_req)

        try:
            response_data = await asyncio.wait_for(queued_req.future, timeout=REQUEST_TIMEOUT_SECONDS)
            return response_data
        except asyncio.TimeoutError:
            logging.error(f"Chat request timed out for model {model_id} after {REQUEST_TIMEOUT_SECONDS} seconds.")
            raise HTTPException(status_code=504, detail="Request processing timed out.")
        except Exception as e:
            logging.error(f"Error processing chat request from queue for model {model_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

# Further endpoints and application logic will be added here. 

# --- Batch Processing Worker ---
async def batch_processing_worker():
    global model_registry
    logging.info(f"Batch processing worker starting. MAX_BATCH_SIZE={MAX_BATCH_SIZE}, BATCH_TIMEOUT={BATCH_TIMEOUT}s")

    # This worker relies on the model being loaded by the startup event.
    # It will fetch the model details once it starts its processing loop or if the model_id changes.
    model = None
    tokenizer = None
    model_id_from_args = current_server_args.model_path # Get the configured model path

    # Try to get model/tokenizer initially. If not available, will try again in loop.
    model_record_initial = model_registry.get(model_id_from_args)
    if model_record_initial and model_record_initial.model_instance and model_record_initial.tokenizer_instance:
        model = model_record_initial.model_instance
        tokenizer = model_record_initial.tokenizer_instance
        logging.info(f"Batch worker successfully retrieved model '{model_id_from_args}'.")
    else:
        logging.warning(f"Model '{model_id_from_args}' not immediately available in registry for batch worker. Will retry.")


    while True:
        if not model or not tokenizer: # Periodically check if model got loaded
            model_record_check = model_registry.get(model_id_from_args)
            if model_record_check and model_record_check.model_instance and model_record_check.tokenizer_instance:
                model = model_record_check.model_instance
                tokenizer = model_record_check.tokenizer_instance
                logging.info(f"Batch worker successfully retrieved model '{model_id_from_args}' after delay.")
            else:
                logging.debug("Batch worker: Model still not available. Waiting...")
                await asyncio.sleep(1) # Wait a bit before retrying
                continue # Skip to next iteration to re-check for model

        batch_to_process: List[QueuedRequest] = []
        first_request_received_time = None

        try:
            # Wait for the first request
            try:
                if not batch_to_process: # Only wait if batch is empty
                    queued_item = await asyncio.wait_for(REQUEST_QUEUE.get(), timeout=BATCH_TIMEOUT)
                    batch_to_process.append(queued_item)
                    REQUEST_QUEUE.task_done() # Mark as processed from queue perspective
                    first_request_received_time = asyncio.get_event_loop().time()
            except asyncio.TimeoutError:
                # This timeout means no request arrived to start a batch.
                # Continue to the start of the loop to re-check for model and then wait again.
                continue

            # Try to gather more requests for the batch if the first one was received
            if first_request_received_time:
                while len(batch_to_process) < MAX_BATCH_SIZE:
                    elapsed_time = asyncio.get_event_loop().time() - first_request_received_time
                    remaining_time_for_batch_window = BATCH_TIMEOUT - elapsed_time
                    
                    if remaining_time_for_batch_window <= 0:
                        break # Batch window timeout reached

                    try:
                        # Poll queue with a very short timeout to see if more items are immediately available
                        # or up to remaining_time_for_batch_window
                        queued_item = await asyncio.wait_for(REQUEST_QUEUE.get(), timeout=min(0.001, remaining_time_for_batch_window))
                        batch_to_process.append(queued_item)
                        REQUEST_QUEUE.task_done()
                    except asyncio.TimeoutError:
                        break # No more items arrived within the short poll or window ended
                    except asyncio.QueueEmpty: # Should be caught by TimeoutError with wait_for
                        break


            if not batch_to_process:
                continue # Should not happen if logic above is correct, but as a safeguard

            logging.info(f"Processing batch of {len(batch_to_process)} requests.")

            prompts_for_batch: List[str] = []
            request_types_in_batch: List[str] = [] # "completion" or "chat_completion"
            # Store original request objects to correctly form responses later
            original_requests_in_batch: List[QueuedRequest] = list(batch_to_process)


            # --- Parameter Consolidation (Simplified for Step A) ---
            # Use parameters from the first request in the batch.
            # This is a simplification; future steps might need more sophisticated handling.
            first_req_data_for_params = batch_to_process[0].request_data
            
            # Common defaults, overridden by first request if present
            max_tokens = 100 
            temp = 0.7

            if isinstance(first_req_data_for_params, CompletionRequest):
                max_tokens = first_req_data_for_params.max_tokens if first_req_data_for_params.max_tokens is not None else max_tokens
                temp = first_req_data_for_params.temperature if first_req_data_for_params.temperature is not None else temp
            elif isinstance(first_req_data_for_params, ChatCompletionRequest):
                max_tokens = first_req_data_for_params.max_tokens if first_req_data_for_params.max_tokens is not None else max_tokens
                temp = first_req_data_for_params.temperature if first_req_data_for_params.temperature is not None else temp
            # Add other parameters like top_p, etc., if they become part of batching strategy
            
            # --- Prompt Preparation & Expansion for 'n' parameter ---
            expanded_prompts_for_batch: List[str] = []
            expanded_request_types: List[str] = []
            # Maps each item in expanded_prompts_for_batch back to its original QueuedRequest and original request_data
            map_expanded_to_original_qr_and_data: List[Tuple[QueuedRequest, Union[CompletionRequest, ChatCompletionRequest]]] = []


            for qr in batch_to_process: # These are unique QueuedRequest objects from the current batch
                prompt_text_for_req = None
                req_type_for_req = None
                request_data = qr.request_data # Original CompletionRequest or ChatCompletionRequest
                
                # Determine the number of choices ('n') requested
                num_choices = 1 # Default to 1
                if hasattr(request_data, 'n') and request_data.n is not None:
                    if isinstance(request_data.n, int) and request_data.n > 0:
                        num_choices = request_data.n
                    else:
                        logging.warning(f"Invalid 'n' value ({request_data.n}) for request; defaulting to 1.")
                        # Potentially set an error on the future if 'n' is invalid type or <= 0
                        if not qr.future.done():
                            qr.future.set_exception(ValueError(f"Parameter 'n' must be a positive integer, got {request_data.n}"))
                        continue # Skip this request

                try:
                    if isinstance(request_data, CompletionRequest):
                        prompt_text_for_req = request_data.prompt
                        req_type_for_req = "completion"
                    elif isinstance(request_data, ChatCompletionRequest):
                        # Ensure tokenizer is TokenizerWrapper for apply_chat_template
                        current_tokenizer_instance = tokenizer
                        if not isinstance(current_tokenizer_instance, TokenizerWrapper):
                             # This should ideally not happen if tokenizer is always wrapped, but as a safeguard:
                            logging.warning("Tokenizer in batch worker is not TokenizerWrapper, attempting to wrap.")
                            current_tokenizer_instance = TokenizerWrapper(current_tokenizer_instance)

                        chat_history = [msg.model_dump(exclude_none=True) for msg in request_data.messages]
                        prompt_text_for_req = current_tokenizer_instance.apply_chat_template(
                            chat_history, tokenize=False, add_generation_prompt=True
                        )
                        req_type_for_req = "chat_completion"
                    else:
                        # This should have been caught by QueuedRequest type hint, but defensive check
                        raise TypeError(f"Unsupported request data type: {type(request_data)}")

                    if prompt_text_for_req is not None and req_type_for_req is not None:
                        for _ in range(num_choices): # Expand for 'n'
                            expanded_prompts_for_batch.append(prompt_text_for_req)
                            expanded_request_types.append(req_type_for_req)
                            map_expanded_to_original_qr_and_data.append((qr, request_data))
                    else: # Should not happen if logic above is correct
                         logging.error(f"Prompt or request type preparation failed unexpectedly for {request_data}")
                         if not qr.future.done():
                            qr.future.set_exception(RuntimeError("Internal error during prompt preparation."))
                
                except Exception as e:
                    logging.error(f"Failed to prepare prompt for {request_data}: {e}", exc_info=True)
                    if not qr.future.done():
                        qr.future.set_exception(e)
                    # This qr will be skipped for generation as its prompts won't be added

            if not expanded_prompts_for_batch: # If all requests in the fetched batch failed preparation or batch was empty
                logging.warning("No valid prompts to process in the current batch after preparation.")
                # Ensure any futures from batch_to_process that aren't done yet (e.g. due to n<=0 error) are handled
                for qr_check in batch_to_process:
                    if not qr_check.future.done():
                        # This case implies an issue not caught above, or a request was valid but produced no prompts.
                        # Most failures (invalid 'n', prompt prep error) should set future exception already.
                        # If future is not done, it might be a logic error or a request that num_choices=0 (though filtered by >0).
                        logging.warning(f"Request {qr_check.request_data} yielded no prompts but future not set. Setting error.")
                        qr_check.future.set_exception(RuntimeError("Request could not be processed: no prompts generated."))
                continue # Go to the next iteration of the main while loop

            try:
                # Call the batch_generate_text utility
                logging.info(f"Calling batch_generate_text_util with {len(expanded_prompts_for_batch)} expanded prompts.")
                generated_results_batch: List[Tuple[str, int, int]] = await batch_generate_text_util(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=expanded_prompts_for_batch,
                    max_tokens=max_tokens, # Consolidated from first request in batch
                    temp=temp,             # Consolidated from first request in batch
                )

                # --- Distribute Results ---
                if len(generated_results_batch) != len(expanded_prompts_for_batch):
                    logging.error(f"Mismatch in batch results: expected {len(expanded_prompts_for_batch)}, got {len(generated_results_batch)}")
                    # Set error for all original QRs involved in this batch attempt.
                    # Use a set to avoid duplicate error setting on the same future.
                    involved_original_futures = set(item[0].future for item in map_expanded_to_original_qr_and_data)
                    for fut_err in involved_original_futures:
                        if not fut_err.done():
                            fut_err.set_exception(RuntimeError("Batch generation returned unexpected number of results."))
                    continue # Move to next batch cycle

                current_time = int(time.time())
                model_name_for_response = current_server_args.model_path or "unknown_model"

                # Group choices and usage data by original QueuedRequest future
                # Key: Future, Value: Dict containing list of choices, token counts, original request data
                results_for_futures: Dict[Future, Dict[str, Any]] = defaultdict(lambda: {
                    "choices_data_list": [],
                    "prompt_tokens_overall": 0, # Will be set from the first generation for this QR
                    "completion_tokens_overall": 0,
                    "original_request_data": None,
                    "original_queued_request": None,
                    "first_gen_done_for_tokens": False
                })

                for i, (original_qr, original_req_data_tuple_val) in enumerate(map_expanded_to_original_qr_and_data):
                    # original_req_data_tuple_val is the request_data (CompletionRequest or ChatRequest)
                    if original_qr.future.done(): # Skip if already handled (e.g. by prep error)
                        continue

                    text_result, num_prompt_toks_for_this_gen, num_compl_toks_for_this_gen = generated_results_batch[i]
                    
                    # Determine finish_reason (simplified for now)
                    # TODO: Get this accurately from batch_generate_text_util or calculate based on max_tokens per request
                    # This needs the specific max_tokens of original_req_data_tuple_val, not the consolidated one.
                    # current_max_tokens = original_req_data_tuple_val.max_tokens if original_req_data_tuple_val.max_tokens is not None else max_tokens # Fallback to consolidated
                    # finish_reason_val = "length" if num_compl_toks_for_this_gen >= current_max_tokens else "stop"
                    finish_reason_val = "stop" # Placeholder, as in existing TODO

                    choice_data_dict: Dict[str, Any]
                    if isinstance(original_req_data_tuple_val, CompletionRequest):
                        choice_data_dict = {
                            "text": text_result,
                            "logprobs": None, # Not supported
                            "finish_reason": finish_reason_val,
                        }
                    elif isinstance(original_req_data_tuple_val, ChatCompletionRequest):
                        choice_data_dict = {
                            "message": ChatMessage(role="assistant", content=text_result.strip()),
                            "finish_reason": finish_reason_val,
                        }
                    else: 
                        logging.error(f"Internal error: Unknown request data type {type(original_req_data_tuple_val)} during result processing.")
                        if not original_qr.future.done():
                            original_qr.future.set_exception(TypeError(f"Unknown request data type: {type(original_req_data_tuple_val)}"))
                        continue # Skip this generation result

                    # Aggregate results for the original future
                    fut_data = results_for_futures[original_qr.future]
                    if not fut_data["original_queued_request"]: # First time seeing this future
                        fut_data["original_queued_request"] = original_qr
                        fut_data["original_request_data"] = original_req_data_tuple_val
                    
                    fut_data["choices_data_list"].append(choice_data_dict)
                    if not fut_data["first_gen_done_for_tokens"]:
                        fut_data["prompt_tokens_overall"] = num_prompt_toks_for_this_gen
                        fut_data["first_gen_done_for_tokens"] = True
                    fut_data["completion_tokens_overall"] += num_compl_toks_for_this_gen

                # Now, construct and set final responses for each original request
                for fut, result_package in results_for_futures.items():
                    original_qr_final = result_package["original_queued_request"]
                    if not original_qr_final or original_qr_final.future.done(): # Check if future is valid and not already set
                        continue

                    final_choices_list = []
                    for idx, choice_raw_data_item in enumerate(result_package["choices_data_list"]):
                        choice_raw_data_item["index"] = idx # Set the index for the choice
                        if isinstance(result_package["original_request_data"], CompletionRequest):
                            final_choices_list.append(CompletionChoice(**choice_raw_data_item))
                        elif isinstance(result_package["original_request_data"], ChatCompletionRequest):
                            final_choices_list.append(ChatCompletionChoice(**choice_raw_data_item))
                    
                    prompt_tokens_val = result_package["prompt_tokens_overall"]
                    completion_tokens_val = result_package["completion_tokens_overall"]
                    usage_obj = CompletionUsage(
                        prompt_tokens=prompt_tokens_val,
                        completion_tokens=completion_tokens_val,
                        total_tokens=prompt_tokens_val + completion_tokens_val,
                    )

                    response_obj: Union[CompletionResponse, ChatCompletionResponse]
                    req_id_base = f"req_{uuid.uuid4().hex}" # Generic request ID base

                    if isinstance(result_package["original_request_data"], CompletionRequest):
                        response_obj = CompletionResponse(
                            id=f"cmpl-{req_id_base[:29]}",
                            object="text_completion",
                            created=current_time,
                            model=model_name_for_response,
                            choices=final_choices_list,
                            usage=usage_obj,
                        )
                    elif isinstance(result_package["original_request_data"], ChatCompletionRequest):
                        response_obj = ChatCompletionResponse(
                            id=f"chatcmpl-{req_id_base[:28]}",
                            object="chat.completion",
                            created=current_time,
                            model=model_name_for_response,
                            choices=final_choices_list,
                            usage=usage_obj,
                        )
                    else: # Should not happen
                        logging.error(f"Internal error: Unknown original_request_data type {type(result_package['original_request_data'])} for future {fut}")
                        if not original_qr_final.future.done():
                           original_qr_final.future.set_exception(TypeError("Internal error processing response type."))
                        continue
                    
                    if not original_qr_final.future.done():
                        original_qr_final.future.set_result(response_obj)
            
            except Exception as e:
                logging.error(f"Error during batch generation or result distribution: {e}", exc_info=True)
                # Set error for all original QRs involved in this batch attempt whose futures are not yet done.
                involved_original_futures_on_error = set(item[0].future for item in map_expanded_to_original_qr_and_data)
                for fut_err_หนัก in involved_original_futures_on_error:
                    if not fut_err_หนัก.done():
                        fut_err_หนัก.set_exception(e) # Propagate the error

        except Exception as e:
            logging.error(f"Outer error in batch_processing_worker loop: {e}", exc_info=True)

# For debugging or direct execution, if needed (uvicorn main:app --reload)
# if __name__ == "__main__":
#     # This part is typically handled by uvicorn or a similar ASGI server runner.
#     # Not standard to include in the app file itself for production.
#     # For local testing, ensure cli args are somehow mockable or defaults are usable.
#     # Uvicorn would be run from the command line, e.g.:
#     # uvicorn mlx_parallm.server.main:app --host 0.0.0.0 --port 8000 --reload
#     pass 
