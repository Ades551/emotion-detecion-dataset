from dataclasses import dataclass, field

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

    text: str = ""

    # eg. {"llm": {"emotion": "fear", "sentiment": "negative", "score": 1.0}, ...}
    emotion: dict[str, str | float] = field(init=False, default_factory=dict[str, str | float])

    @property
    def duration(self) -> float:
        return self.end - self.start