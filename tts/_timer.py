import time
from dataclasses import dataclass


@dataclass
class Timer:
    time_llm_request: float = -1.0
    time_first_llm_token: float = -1.0
    time_last_llm_token: float = -1.0
    time_first_synthesis_request: float = -1.0
    time_first_audio: float = -1.0
    time_last_audio: float = -1.0

    _num_tokens: int = 0

    @staticmethod
    def _get_time() -> float:
        return time.perf_counter()

    def log_time_llm_request(self) -> None:
        self.time_llm_request = self._get_time()

    def maybe_log_time_first_llm_token(self) -> None:
        if self.time_first_llm_token == -1.0:
            self.time_first_llm_token = self._get_time()

    def maybe_log_time_first_synthesis_request(self) -> None:
        if self.time_first_synthesis_request == -1.0:
            self.time_first_synthesis_request = self._get_time()

    def log_time_first_synthesis_request(self) -> None:
        self.time_first_synthesis_request = self._get_time()

    def log_time_last_llm_token(self) -> None:
        self.time_last_llm_token = self._get_time()

    def maybe_log_time_first_audio(self) -> None:
        if self.time_first_audio == -1.0:
            self.time_first_audio = self._get_time()

    def log_time_last_audio(self) -> None:
        self.time_last_audio = self._get_time()

    def increment_num_tokens(self) -> None:
        self._num_tokens += 1

    def num_seconds_to_first_audio(self) -> float:
        return self.time_first_audio - self.time_first_llm_token

    def num_seconds_to_first_token(self) -> float:
        return self.time_first_llm_token - self.time_llm_request

    def num_seconds_tts_request_first_audio(self) -> float:
        return self.time_first_audio - self.time_first_synthesis_request

    def num_seconds_total_delay(self) -> float:
        return self.num_seconds_to_first_audio() + self.num_seconds_to_first_token()

    def num_tokens_per_second(self) -> float:
        return self._num_tokens / (self.time_last_llm_token - self.time_first_llm_token)

    def wait_for_first_audio(self) -> None:
        while self.time_first_audio == -1.0:
            time.sleep(0.01)

    def wait_for_last_audio(self) -> None:
        while self.time_last_audio == -1.0:
            time.sleep(0.01)

    def reset(self) -> None:
        self.time_llm_request = -1.0
        self.time_first_llm_token = -1.0
        self.time_last_llm_token = -1.0
        self.time_first_synthesis_request = -1.0
        self.time_first_audio = -1.0
        self.time_last_audio = -1.0

        self._num_tokens = 0


__all__ = [
    "Timer",
]
