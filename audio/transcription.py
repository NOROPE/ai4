"""
Transcription buffering and logging.
Accumulates streamed fragments per turn and flushes a single complete line
to the console (INFO) and to the transcription file (TRANSCRIPTION level).
"""

import logging

TRANSCRIPTION_LEVEL = 25  # must match main.py


class TranscriptionBuffer:
    """
    Collects text fragments for a single speaker turn.
    Call `append()` for each fragment and `flush()` at turn end.
    """

    def __init__(self, speaker: str, logger: logging.Logger) -> None:
        self.speaker = speaker
        self._logger = logger
        self._parts: list[str] = []

    def append(self, text: str) -> None:
        text = text.strip()
        if text:
            self._parts.append(text)

    def flush(self) -> None:
        if not self._parts:
            return
        full_text = " ".join(self._parts).strip()
        self._parts.clear()
        if not full_text:
            return
        line = f"{self.speaker}: {full_text}"
        # Single log call: console shows it via the root handler,
        # file handler passes it through TranscriptionOnlyFilter
        self._logger.log(TRANSCRIPTION_LEVEL, line)
