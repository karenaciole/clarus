import sys
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Modelo falso que aceita `bind_tools`, para testar o agente sem provedor real."""

    prompts: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
