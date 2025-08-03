from typing import (
    Any,
    Callable,
    Literal,
)

from datasets import Dataset
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam as ChatMessage,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,  # noqa: F401
)
from openai.types.chat.chat_completion_role import ChatCompletionRole  # noqa: F401
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,  # noqa: F401
)
from openai.types.completion import Completion
from openai.types.shared_params import (  # noqa: F401
    FunctionDefinition,
    FunctionParameters,
)
from pydantic import BaseModel, Field

# typing aliases
MessageType = Literal["chat", "completion"]
ModelResponse = Completion | ChatCompletion | None


Message = str | ChatMessage
Messages = str | list[ChatMessage]
Info = dict[str, Any]
State = dict[str, Any]
SamplingArgs = dict[str, Any]
RewardFunc = Callable[..., float]

# oai tools
JsonPrimitive = Literal["string", "number", "integer", "boolean", "array", "object"]


class GenerateInputs(BaseModel):
    """Pydantic model for generation inputs."""

    prompt: list[Messages]
    answer: list[str] | None = None
    info: list[dict] | None = None
    task: list[str] | None = None
    completion: list[Messages] | None = None


class RolloutRequest(BaseModel):
    """Pydantic model for rollout requests."""

    prompt: Messages
    answer: str = ""
    task: str = "default"
    info: Info = Field(default_factory=dict)

    @classmethod
    def from_dataset(
        cls,
        ds: Dataset,
        *,
        prompt_col: str = "prompt",
        answer_col: str = "answer",
        task_col: str = "task",
        info_col: str = "info",
    ) -> list["RolloutRequest"]:
        """
        Convert a Hugging Face `Dataset` row-wise into RolloutRequest objects.
        Adjust the column names with the keyword arguments if your dataset differs.
        """

        # TODO: revisit this implementation
        # results_dict = {}
        #     if isinstance(inputs, Dataset):
        #         # get prompt column
        #         results_dict = {}
        #         for col in inputs.column_names:
        #             if col == "info":
        #                 # handle info column to ensure mutable dicts
        #                 results_dict[col] = [dict(item) for item in inputs[col]]
        #             else:
        #                 results_dict[col] = deepcopy(inputs[col])
        # ☆ keep the heavy logic here (parsing JSON, defaulting, etc.)
        requests: list[RolloutRequest] = []
        for row in ds:
            requests.append(
                cls(
                    prompt=row[prompt_col],
                    answer=row.get(answer_col, ""),
                    task=row.get(task_col, "default"),
                    info=row.get(info_col, {}) or {},
                )
            )
        return requests

        # preprocess dataset or GenerateInputs to GenerateOutputs
        # results_dict = {}
        # if isinstance(inputs, Dataset):
        #     # get prompt column
        #     results_dict = {}
        #     for col in inputs.column_names:
        #         if col == "info":
        #             # handle info column to ensure mutable dicts
        #             results_dict[col] = [dict(item) for item in inputs[col]]
        #         else:
        #             results_dict[col] = deepcopy(inputs[col])
        # else:
        #     results_dict = {col: deepcopy(inputs[col]) for col in inputs}
        # if "prompt" not in results_dict:
        #     raise ValueError("prompt column not found in inputs")
        # if "answer" not in results_dict and "info" not in results_dict:
        #     raise ValueError("answer or info column must be found in inputs")
        # if "answer" not in results_dict:
        #     results_dict["answer"] = [""] * len(results_dict["prompt"])
        # if "task" not in results_dict:
        #     results_dict["task"] = ["default"] * len(results_dict["prompt"])
        # if "info" not in results_dict:
        #     results_dict["info"] = [{}] * len(results_dict["prompt"])
        # for i, info in enumerate(results_dict["info"]):
        #     if isinstance(info, str):
        #         info = json.loads(info)
        #     if self.oai_tools and "oai_tools" not in info:
        #         info["oai_tools"] = self.oai_tools


class RolloutResult(BaseModel):
    """Data returned after executing a RolloutRequest."""

    completion: Messages
    state: State


class GenerateOutputs(BaseModel):
    """Pydantic model for generation outputs."""

    prompt: list[Messages]
    completion: list[Messages]
    answer: list[str]
    state: list[State]
    info: list[Info]
    task: list[str]
    reward: list[float] = Field(default_factory=list)
    metrics: dict[str, list[float]] = Field(default_factory=dict)


class RolloutScore(BaseModel):
    """Pydantic model for rollout scores."""

    reward: float
    metrics: dict[str, float] = Field(default_factory=dict)


class RolloutScores(BaseModel):
    """Pydantic model for rubric outputs."""

    reward: list[float]
    metrics: dict[str, list[float]] = Field(default_factory=dict)


class ProcessedOutputs(BaseModel):
    """Pydantic model for processed outputs."""

    prompt_ids: list[list[int]]
    prompt_mask: list[list[int]]
    completion_ids: list[list[int]]
    completion_mask: list[list[int]]
    completion_logprobs: list[list[float]]
    rewards: list[float]
