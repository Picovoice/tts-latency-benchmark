import base64
from enum import Enum
from io import BytesIO
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
import soundfile


class AudioEncodings(Enum):
    BYTES = "bytes"
    INT16 = "int16"
    FILE_BUFFER = "mp3"


class AudioSink:
    def __init__(self, sample_rate: int, encoding: AudioEncodings) -> None:
        self._audio = np.array([])
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._buffer = BytesIO()

    def add(self, data: Any) -> None:
        if self._encoding is AudioEncodings.FILE_BUFFER:
            self._buffer.write(data)
        else:
            self._audio = np.concatenate((self._audio, self._decode_chunk(data)))

    def _decode_chunk(self, data: Any) -> Optional[NDArray]:
        if self._encoding is AudioEncodings.INT16:
            return np.array(data, dtype=np.int16)
        elif self._encoding is AudioEncodings.BYTES:
            return np.frombuffer(BytesIO(data).read(), dtype=np.int16)
        elif self._encoding is AudioEncodings.FILE_BUFFER:
            raise ValueError("Cannot decode chunks of MP3 data")
        else:
            raise ValueError(f"Unsupported encoding: `{self._encoding}`")

    def save(self, path: str) -> None:
        if self._encoding is AudioEncodings.FILE_BUFFER:
            with open(path, "wb") as f:
                f.write(self._buffer.getvalue())
        else:
            soundfile.write(path, self._audio.astype(float) / 32768.0, self._sample_rate)

    def reset(self) -> None:
        self._audio = np.array([])
        self._buffer = BytesIO()
