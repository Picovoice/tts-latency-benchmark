from enum import Enum
from typing import (
    Any,
    Generator,
    AsyncGenerator,
)
from queue import Queue
from threading import Thread

import asyncio


class LLMs(Enum):
    PICOLLM = "picollm"


class LLM:
    SYSTEM_MESSAGE = """
    You are a helpful voice assistant. Users ask questions from different domains such as: 
    flights, food-ordering, hotels, movies, music, restaurant-search, sports.
    Give useful answers in a conversational style.
    """

    def __init__(self, system_message: str = SYSTEM_MESSAGE) -> None:
        self._system_message = system_message
        self._response = ""

    def _append_user_message(self, message: str) -> None:
        raise NotImplementedError()

    def _reset_history(self) -> None:
        raise NotImplementedError()

    def _query(self) -> Generator[str, None, None]:
        raise NotImplementedError(
            f"Method `chat_stream` must be implemented in a subclass of {self.__class__.__name__}")

    def _query_async(self) -> AsyncGenerator[str, None]:
        raise NotImplementedError(
            f"Method `chat_stream` must be implemented in a subclass of {self.__class__.__name__}")

    def query(self, user_input: str) -> Generator[str, None, None]:
        self._append_user_message(user_input)
        response = ""
        for token in self._query():
            yield token
            response += token
        self._response = response
        self._reset_history()

    async def query_async(self, user_input: str) -> AsyncGenerator[str, None]:
        self._append_user_message(user_input)
        response = ""
        async for token in self._query_async():
            if token is not None:
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
            LLMs.PICOLLM: picoLLM,
        }

        if llm_type not in classes:
            raise NotImplementedError(f"Cannot create {cls.__name__} of type `{llm_type.value}`")

        return classes[llm_type](**kwargs)

    def __str__(self) -> str:
        raise NotImplementedError()


class picoLLM(LLM):
    RANDOM_SEED = 7777

    def __init__(
            self,
            access_key: str,
            model_path: str,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        import picollm
        self._client = picollm.create(access_key, model_path)
        self._dialog = self._client.get_dialog(system=LLM.SYSTEM_MESSAGE)

    def _append_user_message(self, message: str) -> None:
        self._dialog.add_human_request(message)

    def _reset_history(self) -> None:
        self._dialog = self._client.get_dialog(system=LLM.SYSTEM_MESSAGE)

    def _query(self) -> Generator[str, None, None]:
        queue = Queue()

        def callback(token: str):
            queue.put(token)
        def thread_main():
            self._client.generate(
                prompt=self._dialog.prompt(),
                seed=picoLLM.RANDOM_SEED,
                stream_callback=callback,
                completion_token_limit=128,
            )
            queue.put(None)

        thread = Thread(target=thread_main)
        thread.start()

        while not queue.empty() or thread.is_alive():
            token = queue.get()
            if token is None:
                break
            yield token
        thread.join()

    async def _query_async(self) -> AsyncGenerator[str, None]:
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def callback(token: str):
            loop.call_soon_threadsafe(queue.put_nowait, token)
        def thread_main():
            self._client.generate(
                prompt=self._dialog.prompt(),
                seed=picoLLM.RANDOM_SEED,
                stream_callback=callback,
                completion_token_limit=128,
            )
            loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = Thread(target=thread_main)
        thread.start()

        while not queue.empty() or thread.is_alive():
            token = await queue.get()
            if token is None:
                break
            yield token
            queue.task_done()
        thread.join()

    def __str__(self) -> str:
        return str(self._client)


__all__ = [
    "LLMs",
    "LLM",
]
