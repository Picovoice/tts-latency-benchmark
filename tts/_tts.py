import threading
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import (
    Any,
    Generator,
    Literal,
    Optional,
)

import pvorca
from openai import OpenAI
from pvorca import OrcaActivationLimitError

from audio import AudioSink, AudioEncodings
from ._timer import Timer


class Synthesizers(Enum):
    OPENAI = "openai"
    PICOVOICE_ORCA = "picovoice_orca"
    ELEVENLABS = "elevenlabs"


class Synthesizer:
    def __init__(
            self,
            sample_rate: int,
            timer: Timer,
            text_streamable: bool = False,
            audio_encoding: Optional[AudioEncodings] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.text_streamable = text_streamable

        self._timer = timer
        self._audio_sink = None
        if audio_encoding is not None:
            self._audio_sink = AudioSink(sample_rate=self.sample_rate, encoding=audio_encoding)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        raise NotImplementedError(
            f"Method `synthesize` must be implemented in a subclass of {self.__class__.__name__}")

    def terminate(self) -> None:
        pass

    def _read_text_stream(self, text_stream: Generator[str, None, None]) -> str:
        text = ""
        for token in text_stream:
            self._timer.maybe_log_time_first_llm_token()
            text += token
            self._timer.increment_num_tokens()

        self._timer.log_time_last_llm_token()

        return text

    def save_and_reset_last_audio(self, path: str) -> None:
        self._audio_sink.save(path)
        self._audio_sink.reset()

    @classmethod
    def create(cls, engine: Synthesizers, **kwargs: Any) -> 'Synthesizer':
        subclasses = {
            Synthesizers.ELEVENLABS: ElevenLabsSynthesizer,
            Synthesizers.PICOVOICE_ORCA: PicovoiceOrcaSynthesizer,
            Synthesizers.OPENAI: OpenAISynthesizer,
        }

        if engine not in subclasses:
            raise NotImplementedError(f"Cannot create {cls.__name__} of type `{engine.value}`")

        return subclasses[engine](**kwargs)

    def __str__(self) -> str:
        raise NotImplementedError()


# https://elevenlabs.io/docs/api-reference/websockets
class ElevenLabsSynthesizer(Synthesizer):
    NAME = "ElevenLabs"

    SAMPLE_RATE = 22050
    AUDIO_ENCODING = AudioEncodings.BASE64

    def __init__(
            self,
            access_key: str,
            **kwargs: Any
    ) -> None:
        super().__init__(sample_rate=self.SAMPLE_RATE, **kwargs)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        raise NotImplementedError()

    @property
    def info(self) -> str:
        return f"{self.NAME}"

    def __str__(self) -> str:
        return f"{self.NAME}"


class OpenAISynthesizer(Synthesizer):
    NAME = "OpenAI TTS"
    DEFAULT_MODEL_NAME = "tts-1"
    DEFAULT_VOICE_NAME = "shimmer"

    SAMPLE_RATE = 24000
    AUDIO_ENCODING = AudioEncodings.BYTES

    def __init__(
            self,
            access_key: str,
            model_name: str = DEFAULT_MODEL_NAME,
            voice_name: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = DEFAULT_VOICE_NAME,
            **kwargs: Any
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs)

        self._model_name = model_name
        self._voice_name = voice_name
        self._client = OpenAI(api_key=access_key)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        text = self._read_text_stream(text_stream)

        self._timer.maybe_log_time_first_synthesis_request()

        response = self._client.audio.speech.create(
            model=self._model_name,
            voice=self._voice_name,
            response_format="pcm",
            input=text)

        for data in response.iter_bytes(chunk_size=1024):
            self._timer.maybe_log_time_first_audio()
            if self._audio_sink is not None:
                self._audio_sink.add(data=data)
            else:
                break
        self._timer.log_time_last_audio()

    def __str__(self) -> str:
        return f"{self.NAME}"


class PicovoiceOrcaSynthesizer(Synthesizer):
    AUDIO_ENCODING = AudioEncodings.INT16

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
            text_streamable=True,
            audio_encoding=self.AUDIO_ENCODING)

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
                self._timer.log_time_last_audio()
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
                if self._audio_sink is not None:
                    self._audio_sink.add(data=pcm)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        for token in text_stream:
            self._timer.maybe_log_time_first_llm_token()
            self._synthesize(token)
            self._timer.increment_num_tokens()

        self._timer.log_time_last_llm_token()

        self._flush()

    def _synthesize(self, text: str) -> None:
        self._queue.put_nowait(self.OrcaTextInput(text=text, flush=False))

    def _flush(self) -> None:
        self._queue.put_nowait(self.OrcaTextInput(text="", flush=True))
        self._close_thread_blocking()
        self._start_thread()

    def terminate(self):
        self._close_thread_blocking()
        self._orca_stream.close()
        self._orca.delete()

    def __str__(self) -> str:
        return "Picovoice Orca"


__all__ = [
    "Synthesizers",
    "Synthesizer",
]
