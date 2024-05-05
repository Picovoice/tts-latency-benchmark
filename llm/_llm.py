from enum import Enum
from typing import (
    Any,
    Generator,
)


class LLMs(Enum):
    OPENAI = "openai"


class LLM:
    SYSTEM_MESSAGE = """
    You are a helpful voice assistant. Users ask questions from different domains such as: 
    flights, food-ordering, hotels, movies, music, restaurant-search, sports.
    Give useful answers in a conversational style.     
    """

    def __init__(self, system_message: str = SYSTEM_MESSAGE) -> None:
        self._system_message = system_message
        self._history = [{"role": "system", "content": self._system_message}]
        self._response = ""

    def _append_user_message(self, message: str) -> None:
        self._history.append({"role": "user", "content": message})

    def _reset_history(self) -> None:
        self._history = [{"role": "system", "content": self._system_message}]

    def _query(self, user_input: str) -> Generator[str, None, None]:
        raise NotImplementedError(
            f"Method `chat_stream` must be implemented in a subclass of {self.__class__.__name__}")

    def query(self, user_input: str) -> Generator[str, None, None]:
        self._append_user_message(user_input)
        response = ""
        for token in self._query(user_input=user_input):
            yield token
            response += token
        self._response = response
        self._reset_history()

    @property
    def last_response(self) -> str:
        return self._response

    @classmethod
    def create(cls, llm_type: LLMs, **kwargs) -> 'LLM':
        classes = {
            LLMs.OPENAI: OpenAILLM,
        }

        if llm_type not in classes:
            raise NotImplementedError(f"Cannot create {cls.__name__} of type `{llm_type.value}`")

        return classes[llm_type](**kwargs)

    def __str__(self) -> str:
        raise NotImplementedError()


class OpenAILLM(LLM):
    MODEL_NAME = "gpt-3.5-turbo"
    RANDOM_SEED = 7777

    def __init__(
            self,
            access_key: str,
            model_name: str = MODEL_NAME,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        from openai import OpenAI
        self._model_name = model_name
        self._client = OpenAI(api_key=access_key)

    def _query(self, user_input: str) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=self._model_name,
            messages=self._history,
            seed=self.RANDOM_SEED,
            stream=True)
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token is not None:
                yield token
        self._reset_history()

    def __str__(self) -> str:
        return f"ChatGPT ({self._model_name})"


__all__ = [
    "LLMs",
    "LLM",
]
