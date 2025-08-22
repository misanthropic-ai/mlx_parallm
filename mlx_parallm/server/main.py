from fastapi import FastAPI, HTTPException, Depends
from starlette.responses import StreamingResponse
from typing import List, Any, AsyncGenerator, Tuple, Optional, Union, Dict
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
from mlx_parallm.utils import apply_chat_template_cached
# Access CLI args dynamically to avoid import-time binding issues
try:
    import mlx_parallm.cli as _cli_mod
except Exception:
    _cli_mod = None  # type: ignore
    logging.warning(
        "Could not import mlx_parallm.cli. If not run via CLI, provide model via API."
    )

app = FastAPI(
    title="mlx_parallm Server",
    version="0.1.0",
    description="A high-performance, parallelized batch generation server for MLX models.",
)

from mlx_parallm.server.state import model_registry

# Lightweight in-process metrics
METRICS: Dict[str, Any] = {
    "batches_processed": 0,
    "batch_fill_acc": 0.0,
    "batch_fill_samples": 0,
    "queue_depth_last": 0,
    "stream_batches_processed": 0,
    # Telemetry: tokens/sec and histogram
    "prompt_tokens_total": 0,
    "prompt_time_total": 0.0,
    "prompt_tps_last": 0.0,
    "decode_tokens_total": 0,
    "decode_time_total": 0.0,
    "decode_tps_last": 0.0,
    # 10 buckets for batch fill %: 0-10, 10-20, ..., 90-100
    "batch_fill_hist": [0] * 10,
}

# Request Queue and Batching Configuration
REQUEST_QUEUE: asyncio.Queue = asyncio.Queue()
MAX_BATCH_SIZE = 8
BATCH_TIMEOUT = 0.1
REQUEST_TIMEOUT_SECONDS = 60.0
DIVERSE_MODE = False

# Streaming concurrency guard to prevent starvation of batch worker
MAX_CONCURRENT_STREAMS = 4
STREAMING_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)

# Queue for co-batched streaming (chat)
STREAM_CHAT_QUEUE: asyncio.Queue = asyncio.Queue()
STREAM_BATCH_TIMEOUT = 0.02
SCHEDULER_MODE = "default"

class StreamQueuedChat:
    def __init__(self, request: ChatCompletionRequest):
        self.request = request
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.id = f"chatcmpl-{uuid.uuid4().hex[:28]}"

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
    # Apply batching/timeouts from CLI if provided
    global MAX_BATCH_SIZE, BATCH_TIMEOUT, REQUEST_TIMEOUT_SECONDS, STREAMING_SEMAPHORE, MAX_CONCURRENT_STREAMS
    # Fetch CLI args dynamically from module to avoid stale binding
    args = getattr(_cli_mod, "current_server_args", None) if _cli_mod else None
    try:
        if args is not None:
            MAX_BATCH_SIZE = int(getattr(args, "max_batch_size", MAX_BATCH_SIZE))
            BATCH_TIMEOUT = float(getattr(args, "batch_timeout", BATCH_TIMEOUT))
            REQUEST_TIMEOUT_SECONDS = float(getattr(args, "request_timeout_seconds", REQUEST_TIMEOUT_SECONDS))
            # Optional arg in future; keep default otherwise
            max_streams = int(getattr(args, "max_concurrent_streams", MAX_CONCURRENT_STREAMS))
            MAX_CONCURRENT_STREAMS = max_streams
            STREAMING_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)
            global SCHEDULER_MODE
            SCHEDULER_MODE = getattr(args, "scheduler", "default")
            # Diverse mode flag from CLI
            global DIVERSE_MODE
            DIVERSE_MODE = bool(getattr(args, "diverse_mode", False))
            logging.info(f"Batch config: MAX_BATCH_SIZE={MAX_BATCH_SIZE}, BATCH_TIMEOUT={BATCH_TIMEOUT}s, REQUEST_TIMEOUT_SECONDS={REQUEST_TIMEOUT_SECONDS}")
            logging.info(f"Streaming concurrency limit: MAX_CONCURRENT_STREAMS={MAX_CONCURRENT_STREAMS}")
            logging.info(f"Scheduler mode: {SCHEDULER_MODE}")
            logging.info(f"Diverse mode: {DIVERSE_MODE}")
    except Exception as e:
        logging.warning(f"Failed to apply CLI batching/timeouts: {e}")

    # Determine initial model from CLI or environment
    env_model = os.getenv("MLX_PARALLM_MODEL") or os.getenv("MODEL_PATH") or os.getenv("MODEL")
    if not args and env_model:
        logging.info(f"Using model from environment: {env_model}")
        # Minimal env-based scheduler overrides
        try:
            mb = os.getenv("MAX_BATCH_SIZE"); bt = os.getenv("BATCH_TIMEOUT"); rts = os.getenv("REQUEST_TIMEOUT_SECONDS"); mcs = os.getenv("MAX_CONCURRENT_STREAMS"); sched = os.getenv("SCHEDULER") or os.getenv("MLX_PARALLM_SCHEDULER")
            if mb: 
                MAX_BATCH_SIZE = int(mb)
            if bt:
                BATCH_TIMEOUT = float(bt)
            if rts:
                REQUEST_TIMEOUT_SECONDS = float(rts)
            if mcs:
                MAX_CONCURRENT_STREAMS = int(mcs); STREAMING_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)
            if sched:
                SCHEDULER_MODE = str(sched)
        except Exception as e:
            logging.warning(f"Failed to apply env config: {e}")

    # Env override for diverse mode (allows 'true', '1')
    try:
        env_div = os.getenv("DIVERSE_MODE") or os.getenv("MLX_PARALLM_DIVERSE")
        if env_div is not None:
            val = str(env_div).strip().lower()
            if val in ("1", "true", "yes", "on"):
                DIVERSE_MODE = True
            elif val in ("0", "false", "no", "off"):
                DIVERSE_MODE = False
            logging.info(f"Diverse mode (env): {DIVERSE_MODE}")
    except Exception:
        pass

    model_source = (args.model_path if (args and getattr(args, 'model_path', None)) else env_model)
    if model_source:
        model_id_cli = model_source # Use path as ID for now
        logging.info(f"Attempting to load initial model from CLI: {model_id_cli}")
        
        record = InternalModelRecord(
            id=model_id_cli,
            path_or_hf_id=model_source,
            status=ModelStatus.LOADING,
            model_type="causal_lm",
            adapter_path=(getattr(args, "lora_path", None) if args else None),
        )
        model_registry[model_id_cli] = record

        try:
            # Actual model loading
            # Load base model and optionally apply LoRA/DoRA adapter if provided
            model_instance, tokenizer_instance = load_model_and_tokenizer_util(
                model_source,
                adapter_path=(getattr(args, "lora_path", None) if args else None),
            )
            
            # Update the record in the registry
            record.model_instance = model_instance
            record.tokenizer_instance = tokenizer_instance
            record.status = ModelStatus.LOADED
            record.adapter_path = (getattr(args, "lora_path", None) if args else None)
            # model_type might be refined here if load_model_from_util provides more info
            if args and getattr(args, "lora_path", None):
                logging.info(
                    f"Successfully loaded model: {model_id_cli} with adapter: {args.lora_path}"
                )
            else:
                logging.info(f"Successfully loaded model: {model_id_cli}")
        except Exception as e:
            record.status = ModelStatus.ERROR_LOADING
            logging.error(f"Failed to load model {model_id_cli}: {e}", exc_info=True)
        
        logging.info(f"Model {model_id_cli} registered with status: {record.status.value}")
    else:
        logging.warning("No initial model path found in server arguments. Model registry will be empty at startup.")

    # Start workers based on scheduler mode
    if SCHEDULER_MODE == "continuous":
        asyncio.create_task(continuous_scheduler_worker())
        logging.info("Continuous scheduler worker task created.")
    else:
        asyncio.create_task(batch_processing_worker())
        logging.info("Batch processing worker task created.")
        asyncio.create_task(streaming_batch_worker())
        logging.info("Streaming batch worker task created.")

@app.get("/health", tags=["General"])
async def health_check():
    """
    Endpoint to check the health of the server.
    Returns a simple status indicating the server is operational.
    """
    return {"status": "ok"}

@app.get("/debug/metrics", tags=["Debug"])
async def debug_metrics():
    try:
        avg_fill = (
            (METRICS["batch_fill_acc"] / METRICS["batch_fill_samples"]) if METRICS["batch_fill_samples"] else 0.0
        )
    except Exception:
        avg_fill = 0.0
    # Compute running average TPS where possible
    try:
        prompt_tps_avg = (
            (METRICS["prompt_tokens_total"] / METRICS["prompt_time_total"]) if METRICS["prompt_time_total"] > 1e-9 else 0.0
        )
    except Exception:
        prompt_tps_avg = 0.0
    try:
        decode_tps_avg = (
            (METRICS["decode_tokens_total"] / METRICS["decode_time_total"]) if METRICS["decode_time_total"] > 1e-9 else 0.0
        )
    except Exception:
        decode_tps_avg = 0.0
    return {
        "batches_processed": METRICS.get("batches_processed", 0),
        "avg_batch_fill_pct": avg_fill,
        "batch_fill_hist": METRICS.get("batch_fill_hist", []),
        "queue_depth_last": METRICS.get("queue_depth_last", 0),
        "stream_batches_processed": METRICS.get("stream_batches_processed", 0),
        "prompt_tps_avg": prompt_tps_avg,
        "prompt_tps_last": METRICS.get("prompt_tps_last", 0.0),
        "decode_tps_avg": decode_tps_avg,
        "decode_tps_last": METRICS.get("decode_tps_last", 0.0),
        "prompt_tokens_total": METRICS.get("prompt_tokens_total", 0),
        "decode_tokens_total": METRICS.get("decode_tokens_total", 0),
    }

@app.get("/debug/metrics", tags=["Debug"])
async def debug_metrics():
    try:
        avg_fill = (
            (METRICS["batch_fill_acc"] / METRICS["batch_fill_samples"]) if METRICS["batch_fill_samples"] else 0.0
        )
    except Exception:
        avg_fill = 0.0
    # Compute running average TPS where possible
    try:
        prompt_tps_avg = (
            (METRICS["prompt_tokens_total"] / METRICS["prompt_time_total"]) if METRICS["prompt_time_total"] > 1e-9 else 0.0
        )
    except Exception:
        prompt_tps_avg = 0.0
    try:
        decode_tps_avg = (
            (METRICS["decode_tokens_total"] / METRICS["decode_time_total"]) if METRICS["decode_time_total"] > 1e-9 else 0.0
        )
    except Exception:
        decode_tps_avg = 0.0
    return {
        "batches_processed": METRICS.get("batches_processed", 0),
        "avg_batch_fill_pct": avg_fill,
        "batch_fill_hist": METRICS.get("batch_fill_hist", []),
        "queue_depth_last": METRICS.get("queue_depth_last", 0),
        "stream_batches_processed": METRICS.get("stream_batches_processed", 0),
        "prompt_tps_avg": prompt_tps_avg,
        "prompt_tps_last": METRICS.get("prompt_tps_last", 0.0),
        "decode_tps_avg": decode_tps_avg,
        "decode_tps_last": METRICS.get("decode_tps_last", 0.0),
        "prompt_tokens_total": METRICS.get("prompt_tokens_total", 0),
        "decode_tokens_total": METRICS.get("decode_tokens_total", 0),
    }

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
        async def limited_gen():
            async with STREAMING_SEMAPHORE:
                async for chunk in _completion_stream_generator(request, request.model, model_instance, tokenizer_instance):
                    yield chunk
        return StreamingResponse(limited_gen(), media_type="text/event-stream")
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
                pi = np.array(probs_i[0].astype(mx.float32))
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
            pi = np.array(probs[0].astype(mx.float32))
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
    tokenizer_instance: TokenizerWrapper,
) -> AsyncGenerator[str, None]:
    """Generates text chunks for streaming chat completion in SSE format."""

    # 1. Prepare the prompt using the chat template
    try:
        messages_for_template = [msg.model_dump(exclude_none=True) for msg in request.messages]
        prompt_text = tokenizer_instance.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        logging.error(f"Error applying chat template for model {model_id} during stream: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat messages for streaming: {str(e)}")

    # Tokenize the prompt(s). batch_stream_generate_text expects tokenized input.
    if tokenizer_instance._tokenizer.pad_token is None:
        tokenizer_instance._tokenizer.pad_token = tokenizer_instance.eos_token
    tokenizer_instance._tokenizer.padding_side = 'left'

    prompts_tokens_list = [prompt_text]
    try:
        encoded_prompts = tokenizer_instance._tokenizer(
            prompts_tokens_list, return_tensors="np", padding=False
        )
        prompts_mx_array = mx.array(encoded_prompts["input_ids"])
    except Exception as e:
        logging.error(f"Error tokenizing prompt for model {model_id} during stream: {e}")
        raise HTTPException(status_code=500, detail=f"Error tokenizing prompt for streaming: {str(e)}")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:28]}"
    first_chunk = True

    generation_kwargs = {
        "temp": request.temperature if request.temperature is not None else 0.7,
        "top_p": request.top_p if request.top_p is not None else 1.0,
    }

    try:
        for batch_deltas_and_reasons in batch_stream_generate_text(
            model_instance,
            tokenizer_instance,
            prompts_mx_array,
            request.max_tokens or 1024,
            **generation_kwargs,
        ):
            text_delta, finish_reason = batch_deltas_and_reasons[0]

            choice_delta = DeltaMessage()
            if first_chunk and text_delta is not None:
                choice_delta.role = "assistant"
                first_chunk = False

            if text_delta is not None:
                choice_delta.content = text_delta

            # Only send a chunk if there's content or it's a finish chunk
            if choice_delta.content or choice_delta.role or finish_reason:
                stream_choice = ChatCompletionStreamChoice(
                    index=0,
                    delta=choice_delta,
                    finish_reason=finish_reason,
                )
                chunk = ChatCompletionChunk(
                    id=request_id,
                    model=model_id,
                    choices=[stream_choice],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            if finish_reason:
                break

    except Exception as e:
        logging.error(f"Error during model generation stream for {model_id}: {e}", exc_info=True)
        error_delta = DeltaMessage(content=f"\n\nError during generation: {str(e)}")
        error_choice = ChatCompletionStreamChoice(index=0, delta=error_delta, finish_reason="error")
        error_chunk = ChatCompletionChunk(id=request_id, model=model_id, choices=[error_choice])
        yield f"data: {error_chunk.model_dump_json()}\n\n"
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
        if request.n is not None and request.n > 1:
            raise HTTPException(status_code=400, detail="Streaming with n > 1 is not currently supported.")

        # Enqueue for co-batched streaming
        queued = StreamQueuedChat(request)
        await STREAM_CHAT_QUEUE.put(queued)

        async def sse_stream():
            async with STREAMING_SEMAPHORE:
                try:
                    while True:
                        chunk = await queued.queue.get()
                        if chunk == "__DONE__":
                            break
                        yield chunk
                finally:
                    # Ensure terminator sent
                    yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")
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
    # Dynamic fetch of CLI args to avoid stale reference
    _args = getattr(_cli_mod, "current_server_args", None) if '_cli_mod' in globals() else None
    model_id_from_args = _args.model_path if (_args and getattr(_args, 'model_path', None)) else None # Get the configured model path

    # Try to get model/tokenizer initially. If not available, will try again in loop.
    model_record_initial = model_registry.get(model_id_from_args) if model_id_from_args else None
    if model_record_initial and model_record_initial.model_instance and model_record_initial.tokenizer_instance:
        model = model_record_initial.model_instance
        tokenizer = model_record_initial.tokenizer_instance
        logging.info(f"Batch worker successfully retrieved model '{model_id_from_args}'.")
    else:
        logging.warning(f"Model '{model_id_from_args}' not immediately available in registry for batch worker. Will retry.")


    while True:
        if not model or not tokenizer: # Periodically check if model got loaded
            model_record_check = model_registry.get(model_id_from_args) if model_id_from_args else None
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
            # Fast path: if queue already has items, drain up to MAX_BATCH_SIZE immediately
            if REQUEST_QUEUE.qsize() > 0:
                first_request_received_time = asyncio.get_event_loop().time()
                try:
                    while len(batch_to_process) < MAX_BATCH_SIZE:
                        item = REQUEST_QUEUE.get_nowait()
                        batch_to_process.append(item)
                        REQUEST_QUEUE.task_done()
                except asyncio.QueueEmpty:
                    pass
            # Slow path: wait up to BATCH_TIMEOUT for the first item
            if not batch_to_process:
                try:
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
                        break  # Batch window timeout reached

                    # Drain immediately available items first without awaiting
                    try:
                        while len(batch_to_process) < MAX_BATCH_SIZE:
                            queued_item = REQUEST_QUEUE.get_nowait()
                            batch_to_process.append(queued_item)
                            REQUEST_QUEUE.task_done()
                    except asyncio.QueueEmpty:
                        pass

                    if len(batch_to_process) >= MAX_BATCH_SIZE:
                        break

                    # Await at most the remaining window for the next item
                    try:
                        queued_item = await asyncio.wait_for(
                            REQUEST_QUEUE.get(), timeout=max(0.0, remaining_time_for_batch_window)
                        )
                        batch_to_process.append(queued_item)
                        REQUEST_QUEUE.task_done()
                    except asyncio.TimeoutError:
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
            top_p = 1.0

            if isinstance(first_req_data_for_params, CompletionRequest):
                max_tokens = first_req_data_for_params.max_tokens if first_req_data_for_params.max_tokens is not None else max_tokens
                temp = first_req_data_for_params.temperature if first_req_data_for_params.temperature is not None else temp
                top_p = first_req_data_for_params.top_p if first_req_data_for_params.top_p is not None else top_p
            elif isinstance(first_req_data_for_params, ChatCompletionRequest):
                max_tokens = first_req_data_for_params.max_tokens if first_req_data_for_params.max_tokens is not None else max_tokens
                temp = first_req_data_for_params.temperature if first_req_data_for_params.temperature is not None else temp
                top_p = first_req_data_for_params.top_p if first_req_data_for_params.top_p is not None else top_p
            # Add other parameters like repetition penalties later if needed
            
            # --- Prompt Preparation & Expansion for 'n' parameter ---
            expanded_prompts_for_batch: List[str] = []
            expanded_request_types: List[str] = []
            # Maps each item in expanded_prompts_for_batch back to its original QueuedRequest and original request_data
            map_expanded_to_original_qr_and_data: List[Tuple[QueuedRequest, Union[CompletionRequest, ChatCompletionRequest]]] = []
            # Track requested n per original future (robust for assembly)
            requested_n_by_future: Dict[Future, int] = defaultdict(int)


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
                        prompt_text_for_req = apply_chat_template_cached(
                            current_tokenizer_instance, chat_history, add_generation_prompt=True
                        )
                        req_type_for_req = "chat_completion"
                    else:
                        # This should have been caught by QueuedRequest type hint, but defensive check
                        raise TypeError(f"Unsupported request data type: {type(request_data)}")

                    if prompt_text_for_req is not None and req_type_for_req is not None:
                        for choice_idx in range(num_choices): # Expand for 'n'
                            # For n>1, add invisible variation to prompt for diversity
                            if num_choices > 1:
                                # Add zero-width spaces as prompt variation for diversity
                                varied_prompt = prompt_text_for_req + "\u200b" * choice_idx
                                expanded_prompts_for_batch.append(varied_prompt)
                                logging.info(f"[DEBUG] Added {choice_idx} zero-width spaces for diversity (choice {choice_idx+1}/{num_choices})")
                            else:
                                expanded_prompts_for_batch.append(prompt_text_for_req)
                            expanded_request_types.append(req_type_for_req)
                            map_expanded_to_original_qr_and_data.append((qr, request_data))
                            requested_n_by_future[qr.future] += 1
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
                try:
                    q_depth = REQUEST_QUEUE.qsize()
                except Exception:
                    q_depth = -1
                if MAX_BATCH_SIZE > 0:
                    fill_pct = (len(expanded_prompts_for_batch) / MAX_BATCH_SIZE) * 100.0
                else:
                    fill_pct = 0.0
                logging.info(f"Batch fill={fill_pct:.1f}% (queue_depth={q_depth})")
                try:
                    METRICS["batches_processed"] += 1
                    METRICS["batch_fill_acc"] += fill_pct
                    METRICS["batch_fill_samples"] += 1
                    METRICS["queue_depth_last"] = int(q_depth)
                    bin_idx = int(min(9, max(0, fill_pct // 10)))
                    METRICS["batch_fill_hist"][int(bin_idx)] += 1
                except Exception:
                    pass
                # Determine whether to skip deduplication
                any_n_gt1 = False
                try:
                    any_n_gt1 = any(
                        (getattr(rd, "n", None) is not None and int(getattr(rd, "n")) > 1)
                        for (_, rd) in map_expanded_to_original_qr_and_data
                    )
                except Exception:
                    any_n_gt1 = False
                
                logging.info(f"[DEBUG] DIVERSE_MODE={DIVERSE_MODE}, any_n_gt1={any_n_gt1}")
                if DIVERSE_MODE or any_n_gt1:
                    # No dedup: process all expanded prompts individually; also disable prefix cache reuse
                    generated_results_batch: List[Tuple[str, int, int]] = await batch_generate_text_util(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=expanded_prompts_for_batch,
                        max_tokens=max_tokens,
                        temp=temp,
                        top_p=top_p,
                        disable_prefix_cache=True,
                    )
                else:
                    # De-duplicate identical prompts within this batch to avoid redundant compute
                    unique_prompts: List[str] = []
                    unique_map: Dict[str, int] = {}
                    positions_for_unique: List[List[int]] = []
                    for idx, ptxt in enumerate(expanded_prompts_for_batch):
                        if ptxt in unique_map:
                            positions_for_unique[unique_map[ptxt]].append(idx)
                        else:
                            unique_map[ptxt] = len(unique_prompts)
                            unique_prompts.append(ptxt)
                            positions_for_unique.append([idx])

                    unique_results: List[Tuple[str, int, int]] = await batch_generate_text_util(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=unique_prompts,
                        max_tokens=max_tokens, # Consolidated from first request in batch
                        temp=temp,             # Consolidated from first request in batch
                        top_p=top_p,
                    )

                    # Expand back to match original ordering
                    generated_results_batch = [None] * len(expanded_prompts_for_batch)  # type: ignore
                    for uidx, positions in enumerate(positions_for_unique):
                        for pos in positions:
                            generated_results_batch[pos] = unique_results[uidx]

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
                _args = getattr(_cli_mod, "current_server_args", None) if _cli_mod else None
                model_name_for_response = _args.model_path if _args else "unknown_model"

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
                    
                    # Determine finish_reason using per-request max_tokens if provided
                    current_max_tokens = (
                        original_req_data_tuple_val.max_tokens
                        if getattr(original_req_data_tuple_val, "max_tokens", None) is not None
                        else max_tokens
                    )
                    finish_reason_val = (
                        "length" if num_compl_toks_for_this_gen >= current_max_tokens else "stop"
                    )

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

                    # Respect requested n; pad/trim as needed to ensure choices length matches request
                    # Derive requested_n from how many expansions mapped to this same future (robust even if request.n missing)
                    try:
                        requested_n = int(requested_n_by_future.get(fut, 0))
                        if requested_n <= 0:
                            requested_n = 1
                    except Exception:
                        requested_n = 1

                    choices_raw = list(result_package["choices_data_list"]) if result_package.get("choices_data_list") else []
                    logging.info(f"[DEBUG] requested_n={requested_n}, choices_raw count={len(choices_raw)}")
                    # Pad with last choice if we have fewer than requested_n
                    if len(choices_raw) > 0 and len(choices_raw) < requested_n:
                        last = choices_raw[-1]
                        choices_raw.extend([last] * (requested_n - len(choices_raw)))
                        logging.info(f"[DEBUG] Padded choices to {len(choices_raw)}")
                    # Trim if more than requested_n
                    if requested_n > 0 and len(choices_raw) > requested_n:
                        choices_raw = choices_raw[:requested_n]
                        logging.info(f"[DEBUG] Trimmed choices to {len(choices_raw)}")

                    # Debug logging for n and aggregation
                    try:
                        logging.info(f"Assembling response: requested_n={requested_n}, aggregated_choices={len(choices_raw)}")
                    except Exception:
                        pass

                    final_choices_list = []
                    for idx, choice_raw_data_item in enumerate(choices_raw):
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
async def streaming_batch_worker():
    """Co-batch streaming chat completions to share compute.

    Groups pending streaming chat requests by model, applies chat templates,
    tokenizes together, and streams per-step deltas back to each client's SSE queue.
    """
    global model_registry
    logging.info("Streaming batch worker starting.")
    while True:
        try:
            queued_first: StreamQueuedChat = await STREAM_CHAT_QUEUE.get()
            model_id = queued_first.request.model
            rec = model_registry.get(model_id)
            if not rec or rec.status != ModelStatus.LOADED or not rec.model_instance or not rec.tokenizer_instance:
                # Fail this request
                await queued_first.queue.put("data: {\"error\":{\"message\":\"Model not ready\"}}\n\n")
                await queued_first.queue.put("__DONE__")
                continue

            # Gather compatible requests within the batch window
            batch_requests: List[StreamQueuedChat] = [queued_first]
            start_t = asyncio.get_event_loop().time()
            while len(batch_requests) < MAX_BATCH_SIZE:
                remaining = STREAM_BATCH_TIMEOUT - (asyncio.get_event_loop().time() - start_t)
                if remaining <= 0:
                    break
                try:
                    nxt: StreamQueuedChat = await asyncio.wait_for(STREAM_CHAT_QUEUE.get(), timeout=remaining)
                    if nxt.request.model == model_id:
                        batch_requests.append(nxt)
                    else:
                        # Put back if model mismatch
                        await STREAM_CHAT_QUEUE.put(nxt)
                        break
                except asyncio.TimeoutError:
                    break

            model = rec.model_instance
            tok = rec.tokenizer_instance
            if not isinstance(tok, TokenizerWrapper):
                tok = TokenizerWrapper(tok)

            # Prepare prompts via chat template
            prompts_text: List[str] = []
            for item in batch_requests:
                try:
                    messages_for_template = [m.model_dump(exclude_none=True) for m in item.request.messages]
                    text = apply_chat_template_cached(tok, messages_for_template, add_generation_prompt=True)
                    prompts_text.append(text)
                except Exception as e:
                    await item.queue.put(f"data: {{\"error\":{{\"message\":\"Template error: {str(e)}\"}}}}\n\n")
                    await item.queue.put("__DONE__")
                    # Mark to skip in generation
                    prompts_text.append("")

            if tok._tokenizer.pad_token is None:
                tok._tokenizer.pad_token = tok.eos_token
            tok._tokenizer.padding_side = 'left'
            enc = tok._tokenizer(prompts_text, return_tensors="np", padding=True)
            prompts_mx = mx.array(enc["input_ids"])  # shape (B, T)

            # Consolidate params from the first request
            first_req = batch_requests[0].request
            max_tokens = first_req.max_tokens or 1024
            temperature = first_req.temperature if first_req.temperature is not None else 0.7
            top_p = first_req.top_p if first_req.top_p is not None else 1.0

            # Initialize per-stream ids and first chunk flags
            first_chunk_flags = [True] * len(batch_requests)

            # Drive batch stream and dispatch per-client chunks
            try:
                for step in batch_stream_generate_text(
                    model,
                    tok,
                    prompts_mx,
                    max_tokens,
                    temp=temperature,
                    top_p=top_p,
                ):
                    # step: List[Tuple[Optional[str], Optional[str]]] per sequence
                    for i, (delta_text, finish_reason) in enumerate(step):
                        item = batch_requests[i]
                        choice_delta = DeltaMessage()
                        if first_chunk_flags[i] and delta_text is not None:
                            choice_delta.role = "assistant"
                            first_chunk_flags[i] = False
                        if delta_text is not None:
                            choice_delta.content = delta_text
                        stream_choice = ChatCompletionStreamChoice(
                            index=0,
                            delta=choice_delta,
                            finish_reason=finish_reason,
                        )
                        chunk = ChatCompletionChunk(
                            id=item.id,
                            model=model_id,
                            choices=[stream_choice],
                        )
                        await item.queue.put(f"data: {chunk.model_dump_json()}\n\n")
                # Done; signal end to all
                for item in batch_requests:
                    await item.queue.put("__DONE__")
                try:
                    METRICS["stream_batches_processed"] += 1
                except Exception:
                    pass
            except Exception as e:
                logging.error(f"Error in streaming batch generation: {e}", exc_info=True)
                for item in batch_requests:
                    await item.queue.put(f"data: {{\"error\":{{\"message\":\"{str(e)}\"}}}}\n\n")
                    await item.queue.put("__DONE__")

        except Exception as outer:
            logging.error(f"Streaming batch worker outer error: {outer}", exc_info=True)
            await asyncio.sleep(0.05)


async def continuous_scheduler_worker():
    """Unified admit-on-step scheduler for streaming and non-streaming.

    This initial version admits at step boundaries: it batches all current
    queued requests, runs a step-wise decode loop, and between steps checks
    queues to admit additional sequences by restarting the step loop with
    the expanded batch.
    """
    global model_registry
    logging.info("Continuous scheduler starting.")
    # Determine model
    _args = getattr(_cli_mod, "current_server_args", None) if '_cli_mod' in globals() else None
    model_id = getattr(_args, "model_path", None)
    if not model_id:
        logging.error("No model_path provided for scheduler.")
        return
    while True:
        rec = model_registry.get(model_id)
        if rec and rec.status == ModelStatus.LOADED and rec.model_instance and rec.tokenizer_instance:
            break
        logging.info("Scheduler waiting for model to load...")
        await asyncio.sleep(0.5)

    model = rec.model_instance
    tokenizer = rec.tokenizer_instance
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # Main scheduling loop
    while True:
        try:
            # Gather a batch from both queues
            batch_queued: List[Union[QueuedRequest, StreamQueuedChat]] = []
            start_t = asyncio.get_event_loop().time()
            # Try to prime with one request from either queue
            try:
                # Prefer request queue if available
                if REQUEST_QUEUE.qsize() > 0:
                    item = await asyncio.wait_for(REQUEST_QUEUE.get(), timeout=BATCH_TIMEOUT)
                    batch_queued.append(item)
                    REQUEST_QUEUE.task_done()
                elif STREAM_CHAT_QUEUE.qsize() > 0:
                    item = await asyncio.wait_for(STREAM_CHAT_QUEUE.get(), timeout=BATCH_TIMEOUT)
                    batch_queued.append(item)
                else:
                    # Wait on either
                    item = await asyncio.wait_for(REQUEST_QUEUE.get(), timeout=BATCH_TIMEOUT)
                    batch_queued.append(item)
                    REQUEST_QUEUE.task_done()
            except asyncio.TimeoutError:
                continue

            # Fill within timeout window
            while len(batch_queued) < MAX_BATCH_SIZE:
                remaining = BATCH_TIMEOUT - (asyncio.get_event_loop().time() - start_t)
                if remaining <= 0:
                    break
                # Drain both queues opportunistically
                took = False
                try:
                    item = REQUEST_QUEUE.get_nowait()
                    batch_queued.append(item)
                    REQUEST_QUEUE.task_done()
                    took = True
                except asyncio.QueueEmpty:
                    pass
                try:
                    item2 = STREAM_CHAT_QUEUE.get_nowait()
                    batch_queued.append(item2)
                    took = True
                except asyncio.QueueEmpty:
                    pass
                if not took:
                    # Await next arrival up to remaining
                    try:
                        item = await asyncio.wait_for(REQUEST_QUEUE.get(), timeout=remaining)
                        batch_queued.append(item)
                        REQUEST_QUEUE.task_done()
                    except asyncio.TimeoutError:
                        break

            if not batch_queued:
                continue

            # Build prompts with expansion for n>1 (for completion requests only); streaming entries stay 1:1
            prompts_text: List[str] = []
            idx_map: List[Tuple[str, Any]] = []  # ('completion'|'stream', original obj)
            requested_n_by_future_cs: Dict[Future, int] = defaultdict(int)
            for item in batch_queued:
                if isinstance(item, QueuedRequest):
                    rd = item.request_data
                    # Determine 'n' (choices)
                    n_choices = 1
                    try:
                        if hasattr(rd, 'n') and rd.n is not None and int(rd.n) > 0:
                            n_choices = int(rd.n)
                    except Exception:
                        n_choices = 1
                    if isinstance(rd, CompletionRequest):
                        for j in range(n_choices):
                            p = rd.prompt if n_choices == 1 else (rd.prompt + ("\u200b" * j))
                            prompts_text.append(p)
                            idx_map.append(("completion", item))
                            requested_n_by_future_cs[item.future] += 1
                    elif isinstance(rd, ChatCompletionRequest):
                        msgs = [m.model_dump(exclude_none=True) for m in rd.messages]
                        base_text = apply_chat_template_cached(tokenizer, msgs, add_generation_prompt=True)
                        for j in range(n_choices):
                            p = base_text if n_choices == 1 else (base_text + ("\u200b" * j))
                            prompts_text.append(p)
                            idx_map.append(("completion", item))
                            requested_n_by_future_cs[item.future] += 1
                else:
                    # StreamQueuedChat (no expansion)
                    msgs = [m.model_dump(exclude_none=True) for m in item.request.messages]
                    text = apply_chat_template_cached(tokenizer, msgs, add_generation_prompt=True)
                    prompts_text.append(text)
                    idx_map.append(("stream", item))

            # Tokenize with left padding
            if tokenizer._tokenizer.pad_token is None:
                tokenizer._tokenizer.pad_token = tokenizer.eos_token
            tokenizer._tokenizer.padding_side = 'left'
            enc = tokenizer._tokenizer(prompts_text, return_tensors="np", padding=True)
            prompts_mx = mx.array(enc["input_ids"])  # (B, T)
            # Metrics: batch fill histogram and queue depth
            try:
                qd = (REQUEST_QUEUE.qsize() if hasattr(REQUEST_QUEUE, "qsize") else 0) + (
                    STREAM_CHAT_QUEUE.qsize() if hasattr(STREAM_CHAT_QUEUE, "qsize") else 0
                )
                METRICS["queue_depth_last"] = int(qd)
                if MAX_BATCH_SIZE > 0:
                    fill_pct = (len(prompts_text) / MAX_BATCH_SIZE) * 100.0
                    METRICS["batches_processed"] += 1
                    METRICS["batch_fill_acc"] += fill_pct
                    METRICS["batch_fill_samples"] += 1
                    bin_idx = int(min(9, max(0, fill_pct // 10)))
                    METRICS["batch_fill_hist"][int(bin_idx)] += 1
            except Exception:
                pass

            # Consolidated params
            # Take from first request if available
            temperature = 0.7
            top_p = 1.0
            max_tokens = 64
            first = batch_queued[0]
            if isinstance(first, QueuedRequest):
                rd = first.request_data
                if isinstance(rd, CompletionRequest) or isinstance(rd, ChatCompletionRequest):
                    if rd.temperature is not None:
                        temperature = rd.temperature
                    if rd.top_p is not None:
                        top_p = rd.top_p
                    if getattr(rd, 'max_tokens', None) is not None:
                        max_tokens = int(rd.max_tokens)  # type: ignore
            elif isinstance(first, StreamQueuedChat):
                r = first.request
                temperature = r.temperature if r.temperature is not None else 0.7
                top_p = r.top_p if r.top_p is not None else 1.0
                max_tokens = r.max_tokens or 64

            # Stepwise generation; between steps, check for new admissions
            # For simplicity, rebuild the generator if we admit new sequences
            admitted_new = False
            accum_texts: List[str] = [""] * len(idx_map)
            finished: List[bool] = [False] * len(idx_map)
            # Metrics: prompt/decode TPS
            # Count total prompt tokens from attention mask if available, else from non-pad tokens
            total_prompt_tokens = 0
            try:
                if "attention_mask" in enc:
                    total_prompt_tokens = int(np.sum(enc["attention_mask"]))
                else:
                    pad_id = tokenizer._tokenizer.pad_token_id
                    arr = np.array(enc["input_ids"])  # type: ignore
                    if pad_id is None:
                        total_prompt_tokens = int(arr.size)
                    else:
                        total_prompt_tokens = int((arr != pad_id).sum())
            except Exception:
                total_prompt_tokens = 0
            t0 = time.perf_counter()
            first_step = True
            decode_tokens_accum = 0
            decode_t_start = None
            for step in batch_stream_generate_text(
                model,
                tokenizer,
                prompts_mx,
                max_tokens,
                temp=temperature,
                top_p=top_p,
            ):
                # On first yield, record prompt time and update metrics
                if first_step:
                    try:
                        prompt_time = max(1e-9, time.perf_counter() - t0)
                        METRICS["prompt_tokens_total"] += int(total_prompt_tokens)
                        METRICS["prompt_time_total"] += float(prompt_time)
                        METRICS["prompt_tps_last"] = (
                            (int(total_prompt_tokens) / float(prompt_time)) if total_prompt_tokens > 0 else 0.0
                        )
                    except Exception:
                        pass
                    decode_t_start = time.perf_counter()
                    first_step = False

                # Decode tokens/sec accumulation: count active sequences before processing this step
                try:
                    active_before = sum(1 for f in finished if not f)
                    decode_tokens_accum += int(active_before)
                except Exception:
                    pass
                # Dispatch step deltas
                for i, (delta_text, finish_reason) in enumerate(step):
                    if finished[i]:
                        continue
                    kind, obj = idx_map[i]
                    if kind == "stream":
                        # SSE chunk
                        choice_delta = DeltaMessage()
                        if delta_text is not None:
                            choice_delta.content = delta_text
                        stream_choice = ChatCompletionStreamChoice(index=0, delta=choice_delta, finish_reason=finish_reason)
                        chunk = ChatCompletionChunk(id=obj.id, model=model_id, choices=[stream_choice])
                        await obj.queue.put(f"data: {chunk.model_dump_json()}\n\n")
                        if finish_reason:
                            await obj.queue.put("__DONE__")
                            finished[i] = True
                    else:
                        if delta_text is not None:
                            accum_texts[i] += delta_text
                        if finish_reason:
                            finished[i] = True

                # Admission check
                if REQUEST_QUEUE.qsize() > 0 or STREAM_CHAT_QUEUE.qsize() > 0:
                    admitted_new = True
                    # Update decode TPS for this segment before rebuilding
                    try:
                        if decode_t_start is not None:
                            dt = max(1e-9, time.perf_counter() - decode_t_start)
                            METRICS["decode_tokens_total"] += int(decode_tokens_accum)
                            METRICS["decode_time_total"] += float(dt)
                            METRICS["decode_tps_last"] = (
                                (int(decode_tokens_accum) / float(dt)) if decode_tokens_accum > 0 else 0.0
                            )
                    except Exception:
                        pass
                    break

            # Resolve non-streaming completions with aggregation for n>1
            current_time = int(time.time())
            # Group generated texts by future
            agg_texts: Dict[Future, List[str]] = defaultdict(list)
            fut_to_obj: Dict[Future, QueuedRequest] = {}
            for i, (kind, obj) in enumerate(idx_map):
                if kind != "completion" or not isinstance(obj, QueuedRequest):
                    continue
                fut_to_obj[obj.future] = obj
                text_i = accum_texts[i]
                agg_texts[obj.future].append(text_i)

            for fut, texts in agg_texts.items():
                obj = fut_to_obj.get(fut)
                if obj is None:
                    continue
                rd = obj.request_data
                model_name_for_response = model_id
                # Derive requested_n
                requested_n = int(requested_n_by_future_cs.get(fut, 0))
                if requested_n <= 0:
                    requested_n = 1
                # Pad/trim
                if len(texts) > requested_n:
                    texts = texts[:requested_n]
                elif len(texts) < requested_n:
                    if len(texts) > 0:
                        texts = texts + [texts[-1]] * (requested_n - len(texts))
                # Build choices list
                usage = CompletionUsage(prompt_tokens=0, completion_tokens=sum(len(t.split()) for t in texts), total_tokens=0)
                if isinstance(rd, CompletionRequest):
                    choices = [CompletionChoice(text=t, index=i, finish_reason="stop") for i, t in enumerate(texts)]
                    resp = CompletionResponse(id=f"cmpl-{uuid.uuid4().hex[:29]}", object="text_completion", created=current_time, model=model_name_for_response, choices=choices, usage=usage)
                else:
                    chs = [ChatCompletionChoice(index=i, message=ChatMessage(role="assistant", content=t.strip()), finish_reason="stop") for i, t in enumerate(texts)]
                    resp = ChatCompletionResponse(id=f"chatcmpl-{uuid.uuid4().hex[:28]}", object="chat.completion", created=current_time, model=model_name_for_response, choices=chs, usage=usage)
                if not obj.future.done():
                    obj.future.set_result(resp)

            # If we admitted new, loop continues to rebuild batch including new ones
            # If we consumed to exhaustion without admit, finalize decode TPS segment
            try:
                if not admitted_new and decode_t_start is not None:
                    dt = max(1e-9, time.perf_counter() - decode_t_start)
                    METRICS["decode_tokens_total"] += int(decode_tokens_accum)
                    METRICS["decode_time_total"] += float(dt)
                    METRICS["decode_tps_last"] = (
                        (int(decode_tokens_accum) / float(dt)) if decode_tokens_accum > 0 else 0.0
                    )
            except Exception:
                pass
        except Exception as e:
            logging.error(f"Continuous scheduler error: {e}", exc_info=True)
            await asyncio.sleep(0.05)
