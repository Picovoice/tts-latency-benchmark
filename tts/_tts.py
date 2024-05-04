import threading
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import (
    Any,
    Literal,
    Optional,
)

import pvorca
from openai import OpenAI
from pvorca import OrcaActivationLimitError

from ._timer import Timer


class Synthesizers(Enum):
    OPENAI = "openai"
    PICOVOICE_ORCA = "picovoice_orca"


class Synthesizer:
    def __init__(
            self,
            sample_rate: int,
            timer: Timer,
            text_streamable: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.text_streamable = text_streamable

        self._timer = timer

    def synthesize(self, text: str) -> None:
        raise NotImplementedError(
            f"Method `synthesize` must be implemented in a subclass of {self.__class__.__name__}")

    @property
    def info(self) -> str:
        raise NotImplementedError(
            f"Method `info` must be implemented in a subclass of {self.__class__.__name__}")

    def flush(self) -> None:
        pass

    def terminate(self) -> None:
        pass

    @classmethod
    def create(cls, engine: Synthesizers, **kwargs: Any) -> 'Synthesizer':
        subclasses = {
            Synthesizers.PICOVOICE_ORCA: PicovoiceOrcaSynthesizer,
            Synthesizers.OPENAI: OpenAISynthesizer,
        }

        if engine not in subclasses:
            raise NotImplementedError(f"Cannot create {cls.__name__} of type `{engine.value}`")

        return subclasses[engine](**kwargs)

    def __str__(self) -> str:
        raise NotImplementedError()


class OpenAISynthesizer(Synthesizer):
    SAMPLE_RATE = 24000
    NAME = "OpenAI TTS"

    DEFAULT_MODEL_NAME = "tts-1"
    DEFAULT_VOICE_NAME = "shimmer"

    def __init__(
            self,
            access_key: str,
            model_name: str = DEFAULT_MODEL_NAME,
            voice_name: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = DEFAULT_VOICE_NAME,
            **kwargs: Any
    ) -> None:
        super().__init__(sample_rate=self.SAMPLE_RATE, **kwargs)

        self._model_name = model_name
        self._voice_name = voice_name
        self._client = OpenAI(api_key=access_key)

    def synthesize(self, text: str) -> None:
        self._timer.maybe_log_time_first_synthesis_request()

        response = self._client.audio.speech.create(
            model=self._model_name,
            voice=self._voice_name,
            response_format="pcm",
            input=text)

        for _ in response.iter_bytes(chunk_size=1024):
            self._timer.maybe_log_time_first_audio()
            break

    @property
    def info(self) -> str:
        return f"{self.NAME} (model: {self.DEFAULT_MODEL_NAME}, voice: {self.DEFAULT_VOICE_NAME})"

    def __str__(self) -> str:
        return f"{self.NAME}"


class PicovoiceOrcaSynthesizer(Synthesizer):
    NUM_TOKENS_PER_PCM_CHUNK = 4

    @dataclass
    class OrcaTextInput:
        text: str
        flush: bool

    def __init__(
            self,
            timer: Timer,
            access_key: str,
            model_path: Optional[str] = None,
            library_path: Optional[str] = None,
    ) -> None:
        self._orca = pvorca.create(access_key=access_key, model_path=model_path, library_path=library_path)
        super().__init__(
            sample_rate=self._orca.sample_rate,
            timer=timer,
            text_streamable=True)

        self._orca_stream = self._orca.open_stream()

        self._queue: Queue[Optional[PicovoiceOrcaSynthesizer.OrcaTextInput]] = Queue()

        self._num_tokens = 0

        self._thread = None
        self._start_thread()

    def _start_thread(self) -> None:
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _close_thread_blocking(self):
        self._queue.put_nowait(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            orca_input = self._queue.get()
            if orca_input is None:
                break

            self._timer.maybe_log_time_first_synthesis_request()

            try:
                if not orca_input.flush:
                    pcm = self._orca_stream.synthesize(orca_input.text)
                else:
                    pcm = self._orca_stream.flush()
            except OrcaActivationLimitError:
                raise ValueError("Orca activation limit reached.")

            if pcm is not None:
                self._timer.maybe_log_time_first_audio()

    def synthesize(self, text: str) -> None:
        self._queue.put_nowait(self.OrcaTextInput(text=text, flush=False))

    def flush(self) -> None:
        self._queue.put_nowait(self.OrcaTextInput(text="", flush=True))
        self._close_thread_blocking()
        self._start_thread()

    def terminate(self):
        self._close_thread_blocking()
        self._orca_stream.close()
        self._orca.delete()

    @property
    def info(self) -> str:
        return f"Picovoice Orca v{self._orca.version}"

    def __str__(self) -> str:
        return "Picovoice Orca"


__all__ = [
    "Synthesizers",
    "Synthesizer",
]
