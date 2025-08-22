from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any, Union, Dict
import time
import uuid
from enum import Enum # Import Enum

# Define ModelStatus Enum
class ModelStatus(str, Enum):
    LOADED = "loaded"
    AVAILABLE_NOT_LOADED = "available_not_loaded"
    ERROR_LOADING = "error_loading"
    LOADING = "loading"

class ModelPermission(BaseModel):
    # Replicating OpenAI's structure, though many fields might be fixed for now
    id: str = Field(default_factory=lambda: f"modelperm-{''.join(random.choices(string.ascii_letters + string.digits, k=12))}") # type: ignore
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False

class ModelCard(BaseModel):
    id: str = Field(..., description="The unique identifier for the model.")
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()), description="Timestamp of when the model was created/loaded.")
    owned_by: str = Field("mlx_parallm", description="The owner of the model (e.g., 'mlx_parallm', 'openai', 'user').")
    # permission: List[ModelPermission] = Field(default_factory=list) # Re-add if needed with proper random id generation
    root: Optional[str] = Field(None, description="The root model ID if this is a fine-tuned model.")
    parent: Optional[str] = Field(None, description="The parent model ID if this is a variant.")
    
    # Custom fields for mlx_parallm
    status: ModelStatus = Field(ModelStatus.AVAILABLE_NOT_LOADED, description="Current status of the model.")
    type: Optional[Literal["causal_lm", "embedding", "classifier", "reward", "general_nn"]] = Field(None, description="Type of the model.")
    path_or_hf_id: Optional[str] = Field(None, description="Filesystem path or Hugging Face ID of the model.")

class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard] = Field(default_factory=list)

# Internal representation, might hold more details or actual model objects later
class InternalModelRecord(BaseModel):
    id: str
    path_or_hf_id: str
    model_type: Optional[Literal["causal_lm", "embedding", "classifier", "reward", "general_nn"]] = None
    status: ModelStatus = ModelStatus.AVAILABLE_NOT_LOADED
    created_timestamp: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mlx_parallm"
    # Track currently applied adapter (e.g., LoRA/DoRA) if any
    adapter_path: Optional[str] = None
    
    # Actual loaded instances (not part of the serialized ModelCard)
    model_instance: Optional[Any] = None
    tokenizer_instance: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True # Allow complex types like model instances

    def to_model_card(self) -> ModelCard:
        return ModelCard(
            id=self.id,
            created=self.created_timestamp,
            owned_by=self.owned_by,
            status=self.status,
            type=self.model_type,
            path_or_hf_id=self.path_or_hf_id
        )

# Need to import these for ModelPermission if re-enabled
# import random
# import string 

# ---- Completion Schemas ----

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: Optional[Any] = None  # Structure: {tokens, token_logprobs, top_logprobs, text_offset}
    finish_reason: Optional[Literal["stop", "length"]] = "stop"

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:29]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Optional[CompletionUsage] = None

class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use for completion.")
    prompt: str = Field(..., description="The prompt(s) to generate completions for.")
    max_tokens: int = Field(100, description="The maximum number of tokens to generate.", ge=0)
    temperature: float = Field(0.0, description="Sampling temperature. 0 means greedy decoding.", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Nucleus sampling parameter.", ge=0.0, le=1.0)
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress.")
    n: Optional[int] = Field(1, description="How many completions to generate for each prompt.")
    logprobs: Optional[int] = Field(None, description="Include the log probabilities on the logprobs most likely tokens.")
    echo: Optional[bool] = Field(False, description="Echo back the prompt in addition to the completion.")
    logit_bias: Optional[Dict[str, float]] = Field(
        None,
        description="Map of token (string or id) to bias added to its logit.")
    # stop: Optional[Union[str, List[str]]] = Field(None, description="Up to 4 sequences where the API will stop generating further tokens.") # For later
    # presence_penalty: float = Field(0.0, ge=-2.0, le=2.0) # For later
    # frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0) # For later
    # user: Optional[str] = None # For later
    # TODO: Add more parameters as supported by mlx_parallm.utils.generate_step or mlx_lm.generate
    # e.g. repetition_penalty, logit_bias

# Schemas for /v1/chat/completions
class ChatMessage(BaseModel):
    role: str # "system", "user", "assistant"
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop" # "stop", "length", "content_filter", "tool_calls" (not yet supported)

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:28]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage
    # system_fingerprint: Optional[str] = None # TODO: Add if we can get this

# ---- Streaming Schemas for Chat Completions ----

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    # tool_calls: Optional[List[Any]] = None # For future tool use support

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None # "stop", "length", "tool_calls", "content_filter"
    # logprobs: Optional[LogProbs] = None # If logprobs are supported in streaming

class ChatCompletionChunk(BaseModel):
    id: str # Should be the same ID as the original ChatCompletionRequest if possible, or a new unique ID for the stream
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str # Model name/ID
    choices: List[ChatCompletionStreamChoice]
    # usage: Optional[CompletionUsage] = None # Usage stats are typically not sent with each chunk
    # system_fingerprint: Optional[str] = None # TODO: Add if we can get this

# Need to import these for ModelPermission if re-enabled or CompletionResponse ID generation
import random
import string 

# ---- Perplexity Schemas ----

class PerplexityRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use for computing perplexity.")
    text: str = Field(..., description="Raw text to evaluate (no chat templating).")

class PerplexityResponse(BaseModel):
    model: str
    token_count: int
    avg_nll: float
    ppl: float
