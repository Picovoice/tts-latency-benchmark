import base64
from enum import Enum
from io import BytesIO
from typing import Any

import numpy as np
from numpy.typing import NDArray
import soundfile


class AudioEncodings(Enum):
    BYTES = "bytes"
    BASE64 = "base64"
    INT16 = "int16"


class AudioSink:
    def __init__(self, sample_rate: int, encoding: AudioEncodings) -> None:
        self._audio = np.array([])
        self._sample_rate = sample_rate
        self._encoding = encoding

    def add(self, data: Any) -> None:
        self._audio = np.concatenate((self._audio, self._decode(data)))

    def _decode(self, data: Any) -> NDArray:
        if self._encoding is AudioEncodings.BASE64:
            raise NotImplementedError("TEST THIS FIRST!")
            # return np.frombuffer(base64.b64decode(data), dtype=np.int16)
        elif self._encoding is AudioEncodings.INT16:
            return np.array(data, dtype=np.int16)
        elif self._encoding is AudioEncodings.BYTES:
            return np.frombuffer(BytesIO(data).read(), dtype=np.int16)
        else:
            raise ValueError(f"Unsupported encoding: `{self._encoding}`")

    def save(self, path: str) -> None:
        soundfile.write(path, self._audio.astype(float) / 32768.0, self._sample_rate)

    def reset(self) -> None:
        self._audio = np.array([])
