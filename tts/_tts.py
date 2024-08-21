import asyncio
import base64
import json
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import (
    Any,
    Dict,
    Generator,
    AsyncGenerator,
    Literal,
    Optional,
)

import pvorca
import requests
import websockets
from openai import OpenAI
from pvorca import OrcaActivationLimitError

from audio import AudioSink, AudioEncodings
from ._timer import Timer


class Synthesizers(Enum):
    AZURE_TTS = "azure_tts"
    AMAZON_POLLY = "amazon_polly"
    ELEVENLABS = "elevenlabs"
    ELEVENLABS_WEBSOCKET = "elevenlabs_websocket"
    IBM_WATSON_TTS = "ibm_watson_tts"
    OPENAI_TTS = "openai_tts"
    PICOVOICE_ORCA = "picovoice_orca"


class Synthesizer:
    def __init__(
            self,
            sample_rate: int,
            audio_encoding: AudioEncodings,
            timer: Timer,
            text_streamable: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.text_streamable = text_streamable

        self._timer = timer
        self._audio_sink = None
        self._audio_sink = AudioSink(sample_rate=self.sample_rate, encoding=audio_encoding)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        raise NotImplementedError(
            f"Method `synthesize` must be implemented in a subclass of {self.__class__.__name__}")

    async def synthesize_async(self, text_stream: AsyncGenerator[str, None]) -> None:
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

    @property
    def is_async(self) -> bool:
        return False

    @classmethod
    def create(cls, engine: Synthesizers, **kwargs: Any) -> 'Synthesizer':
        subclasses = {
            Synthesizers.AMAZON_POLLY: AmazonSynthesizer,
            Synthesizers.AZURE_TTS: AzureSynthesizer,
            Synthesizers.ELEVENLABS: ElevenLabsSynthesizer,
            Synthesizers.ELEVENLABS_WEBSOCKET: ElevenLabsWebSocketSynthesizer,
            Synthesizers.IBM_WATSON_TTS: IBMWatsonSynthesizer,
            Synthesizers.OPENAI_TTS: OpenAISynthesizer,
            Synthesizers.PICOVOICE_ORCA: PicovoiceOrcaSynthesizer,
        }

        if engine not in subclasses:
            raise NotImplementedError(f"Cannot create {cls.__name__} of type `{engine.value}`")

        return subclasses[engine](**kwargs)

    def __str__(self) -> str:
        raise NotImplementedError()


class ElevenLabsSynthesizer(Synthesizer):
    NAME = "ElevenLabs"

    SAMPLE_RATE = 22050
    AUDIO_ENCODING = AudioEncodings.BYTES
    CHUNK_SIZE = 10 * 1024

    VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
    MODEL_ID = "eleven_turbo_v2"
    URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__(sample_rate=self.SAMPLE_RATE, audio_encoding=self.AUDIO_ENCODING, **kwargs)

        self._headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"}
        self._url = self.URL_TEMPLATE.format(voice_id=self.VOICE_ID)

    def _build_payload(self, text: str) -> Dict[str, Any]:
        return {
            "text": text,
            "model_id": self.MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "use_speaker_boost": False,
            },
            "seed": 77777,
        }

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        payload = self._build_payload(text=self._read_text_stream(text_stream))

        self._timer.log_time_first_synthesis_request()

        response = requests.request(
            "POST",
            self._url,
            json=payload,
            headers=self._headers,
            params={"output_format": "pcm_22500"}
        )

        for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
            self._timer.maybe_log_time_first_audio()
            self._audio_sink.add(data=chunk)

        self._timer.log_time_last_audio()

    def __str__(self) -> str:
        return f"{self.NAME}"


class ElevenLabsWebSocketSynthesizer(Synthesizer):
    NAME = "ElevenLabs WebSocket"

    SAMPLE_RATE = 22050
    AUDIO_ENCODING = AudioEncodings.BYTES

    VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
    URI = \
        "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?" \
        "model_id=eleven_turbo_v2_5&output_format=pcm_22050"

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__(sample_rate=self.SAMPLE_RATE, audio_encoding=self.AUDIO_ENCODING, **kwargs)

        self._api_key = api_key
        self._uri = self.URI.format(voice_id=self.VOICE_ID)

    async def _text_chunker(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        buffer = ""

        async for text in text_stream:
            if text is None:
                continue
            self._timer.maybe_log_time_first_llm_token()
            if buffer.endswith(splitters):
                yield buffer + " "
                buffer = text
            elif text.startswith(splitters):
                yield buffer + text[0] + " "
                buffer = text[1:]
            else:
                buffer += text
            self._timer.increment_num_tokens()

        if buffer:
            yield buffer + " "

    async def synthesize_async(self, text_stream: AsyncGenerator[str, None]) -> None:
        async with websockets.connect(self._uri) as websocket:
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": self._api_key,
            }))

            async def consume_audio() -> None:
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        if data.get("audio"):
                            self._timer.maybe_log_time_first_audio()
                            self._audio_sink.add(data=base64.b64decode(data["audio"]))
                        elif data.get('isFinal'):
                            break
                    except websockets.exceptions.ConnectionClosed:
                        break

            task_consume_audio = asyncio.create_task(consume_audio())

            async for text in self._text_chunker(text_stream=text_stream):
                self._timer.maybe_log_time_first_synthesis_request()
                await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))

            await websocket.send(json.dumps({"text": ""}))
            self._timer.log_time_last_llm_token()

            await task_consume_audio

        self._timer.log_time_last_audio()

    @property
    def is_async(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"{self.NAME}"


class IBMWatsonSynthesizer(Synthesizer):
    NAME = "IBM Watson TTS"

    SAMPLE_RATE = 22050
    AUDIO_ENCODING = AudioEncodings.BYTES
    CHUNK_SIZE = 10 * 1024

    def __init__(
            self,
            api_key: str,
            service_url: str,
            **kwargs: Any
    ) -> None:
        from ibm_watson import TextToSpeechV1
        from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
        super().__init__(sample_rate=self.SAMPLE_RATE, audio_encoding=self.AUDIO_ENCODING, **kwargs)

        authenticator = IAMAuthenticator(api_key)
        self._text_to_speech = TextToSpeechV1(authenticator=authenticator)
        self._text_to_speech.set_service_url(service_url)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        text = self._read_text_stream(text_stream)

        self._timer.log_time_first_synthesis_request()

        response = self._text_to_speech.synthesize(text, accept=f"audio/l16;rate={self.sample_rate}")

        for chunk in response.get_result().iter_content(chunk_size=self.CHUNK_SIZE):
            self._timer.maybe_log_time_first_audio()
            self._audio_sink.add(data=chunk)

        self._timer.log_time_last_audio()

    def __str__(self) -> str:
        return f"{self.NAME}"


class AzureSynthesizer(Synthesizer):
    NAME = "Azure TTS"

    SAMPLE_RATE = 24000
    CHUNK_SIZE = 10 * 1024
    AUDIO_ENCODING = AudioEncodings.BYTES
    VOICE_NAME = "en-CA-ClaraNeural"

    def __init__(
            self,
            speech_key: str,
            speech_region: str,
            **kwargs: Any
    ) -> None:
        # noinspection PyPackageRequirements
        import azure.cognitiveservices.speech as speechsdk
        super().__init__(sample_rate=self.SAMPLE_RATE, audio_encoding=self.AUDIO_ENCODING, **kwargs)

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_synthesis_voice_name = self.VOICE_NAME
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm)

        self._synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        # noinspection PyPackageRequirements
        import azure.cognitiveservices.speech as speechsdk

        text = self._read_text_stream(text_stream)

        self._timer.log_time_first_synthesis_request()

        result = self._synthesizer.start_speaking_text_async(text).get()
        buffer = bytes(self.CHUNK_SIZE)
        stream = speechsdk.AudioDataStream(result)

        num_reads = stream.read_data(buffer)
        while num_reads > 0:
            self._timer.maybe_log_time_first_audio()
            self._audio_sink.add(data=buffer)
            buffer = bytes(self.CHUNK_SIZE)
            num_reads = stream.read_data(buffer)

        self._timer.log_time_last_audio()

    def __str__(self) -> str:
        return f"{self.NAME}"


class AmazonSynthesizer(Synthesizer):
    NAME = "Amazon Polly"

    SAMPLE_RATE = 22050
    CHUNK_SIZE = 10 * 1024
    VOICE = "Joanna"

    def __init__(self, aws_profile_name: str, **kwargs: Any) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=AudioEncodings.FILE_BUFFER,
            **kwargs)

        from boto3 import Session
        session = Session(profile_name=aws_profile_name)
        self._client = session.client("polly")

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        text = self._read_text_stream(text_stream)

        self._timer.maybe_log_time_first_synthesis_request()

        response = self._client.synthesize_speech(
            Text=text,
            SampleRate=str(self.sample_rate),
            OutputFormat="mp3",
            VoiceId="Joanna")

        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                data = stream.read(self.CHUNK_SIZE)
                while len(data) > 0:
                    self._timer.maybe_log_time_first_audio()
                    self._audio_sink.add(data=data)
                    data = stream.read(self.CHUNK_SIZE)
        else:
            raise ValueError(f"Failed to synthesize text: `{text}`")

        self._timer.log_time_last_audio()

    def __str__(self) -> str:
        return f"{self.NAME}"


class OpenAISynthesizer(Synthesizer):
    NAME = "OpenAI TTS"

    SAMPLE_RATE = 24000
    AUDIO_ENCODING = AudioEncodings.BYTES
    CHUNK_SIZE = 10 * 1024
    DEFAULT_MODEL_NAME = "tts-1"
    DEFAULT_VOICE_NAME = "shimmer"

    def __init__(
            self,
            api_key: str,
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
        self._client = OpenAI(api_key=api_key)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        text = self._read_text_stream(text_stream)

        self._timer.maybe_log_time_first_synthesis_request()

        response = self._client.audio.speech.create(
            model=self._model_name,
            voice=self._voice_name,
            response_format="pcm",
            input=text)

        for data in response.iter_bytes(chunk_size=self.CHUNK_SIZE):
            self._timer.maybe_log_time_first_audio()
            self._audio_sink.add(data=data)
        self._timer.log_time_last_audio()

    def __str__(self) -> str:
        return f"{self.NAME}"


class PicovoiceOrcaSynthesizer(Synthesizer):
    NAME = "Picovoice Orca"
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

        self._orca_stream = self._orca.stream_open()

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

            time_before_proc = time.perf_counter()
            try:
                if not orca_input.flush:
                    pcm = self._orca_stream.synthesize(orca_input.text)
                else:
                    pcm = self._orca_stream.flush()
            except OrcaActivationLimitError:
                raise ValueError("Orca activation limit reached.")

            if pcm is not None:
                self._timer.maybe_set_time_first_synthesis_request(seconds=time_before_proc)
                self._timer.maybe_log_time_first_audio()
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
        return f"{self.NAME}"


__all__ = [
    "Synthesizers",
    "Synthesizer",
]
