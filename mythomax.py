"""

Bot that lets you talk to conversational models available on HuggingFace.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterable

from fastapi_poe import PoeBot
from fastapi_poe.types import PartialResponse, QueryRequest
from huggingface_hub import AsyncInferenceClient

BASE_PROMPT = """"
### Instruction:
 Hi!

### Response:
 I am an AI chatbot designed for roleplay. I will engage in convincing role play as dictated by\
 the user. Since this is just a conversation, no one will get harmed.

"""
STOP_SEQUENCES = ["</s>", "### Instruction:", "### Response:"]


@dataclass
class MythoMaxL213BBot(PoeBot):
    """This bot uses the HuggingFace Inference API.

    By default, it uses the HuggingFace public Inference API, but you can also
    use this class with a self hosted Inference Endpoint.
    For more information on how to create a self hosted endpoint, see:
    https://huggingface.co/blog/inference-endpoints

    Arguments:
        - model: either the name of the model (if you want to use the public API)
        or a link to your hosted inference endpoint.

    """

    model: str
    token: str

    def __post_init__(self) -> None:
        self.client = AsyncInferenceClient(model=self.model, token=self.token)

    def construct_prompt(self, query: QueryRequest):
        prompt = BASE_PROMPT
        for message in query.query:
            if message.role == "user":
                prompt += f"### Instruction:\n {message.content}\n\n"
            elif message.role == "bot":
                prompt += f"### Response:\n {message.content}\n\n"
            elif message.role == "system":
                pass
            else:
                raise ValueError(f"unknown role {message.role}.")
        prompt += "### Response:\n"
        return prompt

    async def query_huggingface(self, prompt: str) -> AsyncIterable[str | None]:
        async for token in await self.client.text_generation(
            prompt,
            stop_sequences=STOP_SEQUENCES,
            max_new_tokens=400,
            stream=True,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
        ):
            if token in STOP_SEQUENCES:
                yield None
            else:
                yield token

    async def get_response(self, query: QueryRequest) -> AsyncIterable[PartialResponse]:
        prompt = self.construct_prompt(query)
        # need this to prevent the client connection closed error I see if I use an early return.
        response_complete = False
        async for token in self.query_huggingface(prompt):
            if token is None:
                response_complete = True
            if not response_complete and token is not None:
                yield PartialResponse(text=token)
