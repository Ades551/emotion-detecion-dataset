from dataclasses import dataclass, field
from functools import cached_property

@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float

@dataclass
class Segment:
    speaker: str
    start: float
    end: float

    words: list[dict] = field(default_factory=list)

    # eg. {"llm": {"emotion": "fear", "sentiment": "negative", "score": 1.0}, ...}
    emotion: dict[str, str | float] = field(init=False, default_factory=dict[str, str | float])

    @cached_property
    def text(self) -> str:
        return " ".join([w["word"] for w in self.words])

    @property
    def duration(self) -> float:
        return self.end - self.start