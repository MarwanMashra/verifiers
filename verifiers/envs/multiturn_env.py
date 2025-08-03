from abc import abstractmethod
from copy import deepcopy

from openai import AsyncOpenAI

from verifiers.envs.environment import Environment
from verifiers.types import (
    ChatCompletion,
    ChatMessage,
    Completion,
    Messages,
    MessageType,
    RolloutRequest,
    RolloutResult,
    SamplingArgs,
    State,
)


class MultiTurnEnv(Environment):
    def __init__(
        self, message_type: MessageType = "chat", max_turns: int = 10, **kwargs
    ):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.message_type = message_type

    def setup_state(self, state: State, **kwargs) -> State:
        return state

    @abstractmethod
    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        pass

    @abstractmethod
    def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        """
        Generate a response from the environment (messages, state).
        """
        pass

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        request: RolloutRequest,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> RolloutResult:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        is_completed = False
        state = {
            "prompt": request.prompt,
            "completion": [],
            "answer": request.answer,
            "task": request.task,
            "info": request.info,
            "responses": [],
            "turn": 0,
        }
        state = self.setup_state(state)
        if self.message_type == "chat":
            assert isinstance(request.prompt, list)
            completion = []
        else:
            assert isinstance(request.prompt, str)
            completion = ""
        rollout = deepcopy(request.prompt)
        while not is_completed:
            if self.is_completed(rollout, state, **kwargs):
                is_completed = True
                break
            response = await self.get_model_response(
                client=client,
                model=model,
                prompt=rollout,
                oai_tools=request.info.get("oai_tools", None),
                sampling_args=sampling_args,
                message_type=self.message_type,
            )
            state["responses"].append(response)
            if self.message_type == "chat":
                assert isinstance(rollout, list)
                assert isinstance(completion, list)
                assert isinstance(response, ChatCompletion)
                response_text: str = response.choices[0].message.content or ""  # type: ignore
                response_message: ChatMessage = {
                    "role": "assistant",
                    "content": response_text,
                }
                if response.choices[0].message.tool_calls:
                    response_message["tool_calls"] = response.choices[  # type: ignore
                        0
                    ].message.tool_calls
                rollout.append(response_message)
                completion.append(response_message)
            else:
                assert isinstance(rollout, str)
                assert isinstance(completion, str)
                assert isinstance(response, Completion)
                response_text: str = response.choices[0].text or ""  # type: ignore
                rollout += response_text
                completion += response_text
            state["turn"] += 1
            if (
                self.is_completed(rollout, state, **kwargs)
                or state["turn"] >= self.max_turns
            ):
                is_completed = True
            else:
                env_msgs, state = self.env_response(rollout, state, **kwargs)
                if self.message_type == "chat":
                    assert isinstance(env_msgs, list)
                    assert isinstance(rollout, list)
                    assert isinstance(completion, list)
                    rollout += env_msgs
                    completion += env_msgs
                else:
                    assert isinstance(env_msgs, str)
                    assert isinstance(rollout, str)
                    assert isinstance(completion, str)
                    rollout += env_msgs
                    completion += env_msgs

        return RolloutResult(completion=completion, state=state)
