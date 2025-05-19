from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any
import time

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
    status: Literal["loaded", "available_not_loaded", "error_loading", "loading"] = Field("available_not_loaded", description="Current status of the model.")
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
    status: Literal["loaded", "available_not_loaded", "error_loading", "loading"] = "available_not_loaded"
    created_timestamp: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mlx_parallm"
    
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
    logprobs: Optional[Any] = None  # Placeholder for now, OpenAI has a specific LogProbs schema
    finish_reason: Optional[Literal["stop", "length"]] = "stop"

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{''.join(random.choices(string.ascii_letters + string.digits, k=24))}") # type: ignore
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str # ID of the model used
    choices: List[CompletionChoice]
    usage: CompletionUsage
    # system_fingerprint: Optional[str] = None # For later if needed

class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use for completion.")
    prompt: str = Field(..., description="The prompt(s) to generate completions for.")
    max_tokens: int = Field(100, description="The maximum number of tokens to generate.", gt=0)
    temperature: float = Field(0.0, description="Sampling temperature. 0 means greedy decoding.", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Nucleus sampling parameter.", ge=0.0, le=1.0)
    # n: int = Field(1, description="How many completions to generate for each prompt.") # Not supported by current utils.generate
    # stream: bool = Field(False, description="Whether to stream back partial progress.") # For later implementation
    # logprobs: Optional[int] = Field(None, description="Include the log probabilities on the logprobs most likely tokens.") # For later
    # stop: Optional[Union[str, List[str]]] = Field(None, description="Up to 4 sequences where the API will stop generating further tokens.") # For later
    # presence_penalty: float = Field(0.0, ge=-2.0, le=2.0) # For later
    # frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0) # For later
    # user: Optional[str] = None # For later

    # TODO: Add more parameters as supported by mlx_parallm.utils.generate_step or mlx_lm.generate
    # e.g. repetition_penalty, logit_bias

# Need to import these for ModelPermission if re-enabled or CompletionResponse ID generation
import random
import string 