import json
import httpx
import uuid
from smolagents.models import (
    Model,
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallDefinition,
)


class QwenModel(Model):
    def __init__(self, model_id: str, token: str, **kwargs):
        super().__init__(**kwargs)
        self.client = httpx.Client(
            base_url="https://chat.qwenlm.ai/",
            headers={
                "authorization": f"Bearer {token}",
                "user-agent": f"QwenLM Client",
            },
            http2=True,
            timeout=60,
        )
        self.model_id = model_id

    def __call__(
        self,
        messages,
        stop_sequences=None,
        grammar=None,
        tools_to_call_from=None,
        **kwargs,
    ):
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            flatten_messages_as_text=True,
            model=self.model_id,
            stream=False,
            **kwargs,
        )
        completion_kwargs.pop("stop")
        completion_kwargs.pop("tools")
        completion_kwargs.pop("tool_choice")
        response = self.client.post("api/chat/completions", json=completion_kwargs)
        self.client.cookies.clear()
        content = response.json()["choices"][0]["message"]["content"]
        maybe_json = "{" + content.split("{", 1)[-1][::-1].split("}", 1)[-1][::-1] + "}"
        parsed_text: dict = json.loads(maybe_json)
        tool = ChatMessageToolCall(
            id=uuid.uuid4(),
            type="function",
            function=ChatMessageToolCallDefinition(
                name=parsed_text.get("name"),
                arguments=parsed_text.get("arguments"),
            ),
        )

        return ChatMessage(role="assistant", content=content, tool_calls=[tool])
