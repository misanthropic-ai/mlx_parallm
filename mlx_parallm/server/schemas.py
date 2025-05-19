from pydantic import BaseModel, Field
from typing import List, Optional, Literal
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
    # Placeholder for the actual loaded model and tokenizer
    # loaded_model_instance: Any = None 
    # loaded_tokenizer_instance: Any = None

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