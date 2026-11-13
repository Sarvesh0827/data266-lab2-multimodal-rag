from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnRecord:
    question_id: str
    question: str
    answer: str
    status: str


@dataclass
class ConversationMemory:
    conversation_id: str
    assumptions: list[str] = field(default_factory=list)
    session_facts: dict[str, str] = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    last_entities: dict[str, str] = field(default_factory=dict)

    def add_assumption(self, text: str) -> None:
        self.assumptions.append(text)

    def set_fact(self, key: str, value: str) -> None:
        self.session_facts[key] = value

    def add_turn(self, question_id: str, question: str, answer: str, status: str) -> None:
        self.turns.append(TurnRecord(question_id, question, answer, status))

    def context_block(self) -> str:
        parts = []
        if self.assumptions:
            parts.append("Session assumptions (this conversation only):\n- " + "\n- ".join(self.assumptions))
        if self.session_facts:
            facts = ", ".join(f"{k}={v}" for k, v in self.session_facts.items())
            parts.append("Session facts: " + facts)
        if self.turns:
            hist = []
            for t in self.turns[-4:]:
                hist.append(f"Q({t.question_id}): {t.question}\nA: {t.answer} [{t.status}]")
            parts.append("Recent turns:\n" + "\n".join(hist))
        return "\n\n".join(parts)


class MemoryBank:
    """Separate memory per conversation_id; never written to vector DB."""

    def __init__(self):
        self._mem: dict[str, ConversationMemory] = {}

    def get(self, conversation_id: str) -> ConversationMemory:
        if conversation_id not in self._mem:
            self._mem[conversation_id] = ConversationMemory(conversation_id=conversation_id)
        return self._mem[conversation_id]

    def reset(self) -> None:
        self._mem.clear()
