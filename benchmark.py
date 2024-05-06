import argparse
import json
import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib.pyplot as plt
from tqdm import tqdm

from data import TextDatasets, TextDataset
from llm import LLMs, LLM
from tts import (
    Synthesizers,
    Synthesizer,
    Timer,
)

DEFAULT_RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_LLM = LLMs.OPENAI.value
DEBUG = True


@dataclass
class TimingResult:
    num_seconds_first_token: float
    num_seconds_first_audio: float
    num_seconds_total_delay: float
    num_words: int
    num_tokens_per_second: float

    @staticmethod
    def _compute_statistics(results: Sequence['TimingResult'], fn: Callable) -> 'TimingResult':
        return TimingResult(
            num_seconds_total_delay=fn([r.num_seconds_total_delay for r in results]),
            num_seconds_first_token=fn([r.num_seconds_first_token for r in results]),
            num_seconds_first_audio=fn([r.num_seconds_first_audio for r in results]),
            num_words=int(fn([r.num_words for r in results])),
            num_tokens_per_second=fn([r.num_tokens_per_second for r in results]),
        )

    @classmethod
    def mean_from_results(cls, results: Sequence['TimingResult']) -> 'TimingResult':
        if len(results) == 0:
            raise ValueError("Cannot compute mean of empty list")

        def _mean(values: Sequence[Any]) -> float:
            return sum(values) / len(values)

        return cls._compute_statistics(results, _mean)

    @classmethod
    def std_from_results(cls, results: Sequence['TimingResult']) -> 'TimingResult':
        if len(results) == 0:
            raise ValueError("Cannot compute standard deviation of empty list")

        def _std(values):
            mean = sum(values) / len(values)
            return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

        return cls._compute_statistics(results, _std)


class Stats:
    MAX_LLM_DELAY_SECONDS = 0.6

    def __init__(self, tts: Synthesizers, llm: LLMs, results_folder: Optional[str] = None) -> None:
        self._results = []

        self._tts_type_string = tts.value
        self._llm_type_string = llm.value

        self._output_folder = os.path.join(results_folder or DEFAULT_RESULTS_FOLDER, f"llm_{self._llm_type_string}")
        os.makedirs(self._output_folder, exist_ok=True)

    def accumulate(self, timing_result: TimingResult) -> None:
        self._results.append(timing_result)

    def _filter_outliers(self, results: Sequence[TimingResult]) -> Sequence[TimingResult]:
        filtered_results = []
        for result in results:
            if result.num_seconds_first_token > self.MAX_LLM_DELAY_SECONDS:
                continue

            filtered_results.append(result)

        return filtered_results

    def save_results(self) -> None:
        results = self._filter_outliers(self._results)

        num_sentences = len(results)

        mean = TimingResult.mean_from_results(results)
        std = TimingResult.std_from_results(results)

        print("Summary statistics:")
        print(f"Total number of sentences: {num_sentences}")
        print(f"Mean total audio delay: {mean.num_seconds_total_delay:.2f} +- {std.num_seconds_first_audio:.2f} s")
        print(f"Mean delay LLM: {mean.num_seconds_first_token:.2f} +- {std.num_seconds_first_token:.2f} s")
        print(f"Mean delay TTS: {mean.num_seconds_first_audio:.2f} +- {std.num_seconds_first_audio:.2f} s")
        print(f"Mean number of words per sentence: {mean.num_words:.1f} +- {std.num_words:.1f}")
        print(f"Mean tokens per second: {mean.num_tokens_per_second:.2f} +- {std.num_tokens_per_second:.2f}")

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].hist([r.num_seconds_total_delay for r in self._results], bins=10)
        axs[0, 0].set_title('num_seconds_total_delay')
        axs[0, 1].hist([r.num_seconds_first_token for r in self._results], bins=10)
        axs[0, 1].set_title('num_seconds_first_token')
        axs[0, 1].axvline(x=self.MAX_LLM_DELAY_SECONDS, color='r', linestyle='--')
        axs[1, 0].hist([r.num_seconds_first_audio for r in self._results], bins=10)
        axs[1, 0].set_title('num_seconds_first_audio')
        axs[1, 1].hist([r.num_words for r in self._results], bins=10)
        axs[1, 1].set_title('num_words')

        output_path = os.path.join(self._output_folder, f"hists_tts_{self._tts_type_string}.png")
        plt.savefig(output_path)
        plt.close()

        results_json_path = os.path.join(self._output_folder, f"results_tts_{self._tts_type_string}.json")
        results_dict = {
            "total_sentences": num_sentences,
            "mean_total_delay": mean.num_seconds_total_delay,
            "mean_llm_delay": mean.num_seconds_first_token,
            "mean_tts_delay": mean.num_seconds_first_audio,
            "mean_words_per_sentence": mean.num_words,
            "mean_tokens_per_second": mean.num_tokens_per_second,
            "std_total_delay": std.num_seconds_total_delay,
            "std_llm_delay": std.num_seconds_first_token,
            "std_tts_delay": std.num_seconds_first_audio,
            "std_words_per_sentence": std.num_words,
            "std_tokens_per_second": std.num_tokens_per_second,
        }
        with open(results_json_path, "w") as f:
            json.dump(results_dict, f, indent=4)

        print("Results saved to:", self._output_folder)

    @staticmethod
    def load_results(json_path: str) -> Tuple[Synthesizers, TimingResult, TimingResult]:
        tts_type_string = None
        for synthesizer in Synthesizers:
            if synthesizer.value in json_path:
                tts_type_string = synthesizer.value
        if tts_type_string is None:
            raise ValueError(f"Could not determine TTS type from path: `{json_path}`")

        with open(json_path, "r") as f:
            results_dict = json.load(f)

        mean = TimingResult(
            num_seconds_total_delay=results_dict["mean_total_delay"],
            num_seconds_first_token=results_dict["mean_llm_delay"],
            num_seconds_first_audio=results_dict["mean_tts_delay"],
            num_words=results_dict["mean_words_per_sentence"],
            num_tokens_per_second=results_dict["mean_tokens_per_second"],
        )
        std = TimingResult(
            num_seconds_total_delay=results_dict["std_total_delay"],
            num_seconds_first_token=results_dict["std_llm_delay"],
            num_seconds_first_audio=results_dict["std_tts_delay"],
            num_words=results_dict["std_words_per_sentence"],
            num_tokens_per_second=results_dict["std_tokens_per_second"],
        )

        return Synthesizers(tts_type_string), mean, std


def get_llm_init_kwargs(args: argparse.Namespace) -> Dict[str, str]:
    kwargs = dict()
    llm_type = LLMs(args.llm)

    if llm_type is LLMs.OPENAI:
        if args.openai_access_key is None:
            raise ValueError(
                f"An OpenAI access key is required when using OpenAI models. Specify with `--openai-access-key`.")

        kwargs["access_key"] = args.openai_access_key
        if args.system_message is not None:
            kwargs["system_message"] = args.system_message

    return kwargs


def get_synthesizer_init_kwargs(args: argparse.Namespace) -> Dict[str, str]:
    kwargs = dict()
    synthesizer_type = Synthesizers(args.synthesizer)

    if synthesizer_type is Synthesizers.PICOVOICE_ORCA:
        if args.picovoice_access_key is None:
            raise ValueError(
                "Picovoice access key is required when using Picovoice TTS. Specify with `--picovoice-access-key`.")
        kwargs["access_key"] = args.picovoice_access_key
        kwargs["model_path"] = args.orca_model_path
        kwargs["library_path"] = args.orca_library_path

    elif synthesizer_type is Synthesizers.AZURE_TTS:
        if args.azure_speech_key is None or args.azure_speech_region is None:
            raise ValueError(
                "Azure speech key and region are required when using Azure TTS. "
                "Specify with `--azure-speech-key` and `--azure-speech-region`.")
        kwargs['speech_key'] = args.azure_speech_key
        kwargs['speech_region'] = args.azure_speech_region

    elif synthesizer_type is Synthesizers.AMAZON_POLLY:
        if args.aws_profile_name is None:
            raise ValueError(
                "AWS profile name is required when using AWS Polly. Specify with `--aws-profile-name`.")
        kwargs["aws_profile_name"] = args.aws_profile_name

    elif synthesizer_type is Synthesizers.IBM_WATSON_TTS:
        if args.ibm_watson_api_key is None or args.ibm_watson_service_url is None:
            raise ValueError(
                "IBM Watson API key and service URL are required when using IBM Watson TTS. "
                "Specify with `--ibm-watson-api-key` and `--ibm-watson-service-url`.")
        kwargs["api_key"] = args.ibm_watson_api_key
        kwargs["service_url"] = args.ibm_watson_service_url

    elif synthesizer_type is Synthesizers.OPENAI_TTS:
        if args.openai_access_key is None:
            raise ValueError(
                f"An OpenAI access key is required when using OpenAI models. Specify with `--openai-access-key`.")
        kwargs["access_key"] = args.openai_access_key

    return kwargs


def main(args: argparse.Namespace) -> None:
    num_interactions = args.num_interactions
    llm_type = LLMs(args.llm)
    tts_type = Synthesizers(args.synthesizer)
    results_folder = args.results_folder

    dataset = TextDataset.create(TextDatasets.TASKMASTER2)

    timer = Timer()

    synthesizer_init_kwargs = get_synthesizer_init_kwargs(args)
    synthesizer = Synthesizer.create(
        Synthesizers(args.synthesizer),
        timer=timer,
        **synthesizer_init_kwargs)

    llm_init_kwargs = get_llm_init_kwargs(args)
    llm = LLM.create(LLMs(args.llm), **llm_init_kwargs)

    benchmark_sentences = dataset.get_random_sentences(num=num_interactions)

    stats = Stats(llm=llm_type, tts=tts_type, results_folder=results_folder)

    counter = 0
    print("Running benchmark ...")
    for sentence in tqdm(benchmark_sentences):
        timer.reset()

        timer.log_time_llm_request()

        synthesizer.synthesize(text_stream=llm.query(sentence))

        timer.wait_for_first_audio()

        timing_result = TimingResult(
            num_seconds_total_delay=timer.num_seconds_total_delay(),
            num_seconds_first_token=timer.num_seconds_to_first_token(),
            num_seconds_first_audio=timer.num_seconds_to_first_audio(),
            num_words=len(llm.last_response.split()),
            num_tokens_per_second=timer.num_tokens_per_second())
        stats.accumulate(timing_result=timing_result)

        if DEBUG:
            print(f"Input: {sentence}")
            print(f"Answer: {llm.last_response}")
            print(f"llm request -> first token: {timer.num_seconds_to_first_token():.2f}")
            print(f"first token -> first audio: {timer.num_seconds_to_first_audio():.2f}")
            print(f"tts request -> first audio: {timer.num_seconds_tts_request_first_audio():.2f}")
            print(f"Total delay (TTFB - time to first byte): {timer.num_seconds_total_delay():.2f}")
            timer.wait_for_last_audio()
            audio_path = os.path.join(results_folder, f"audio_{counter}.wav")
            synthesizer.save_and_reset_last_audio(audio_path)
            print(f"Saved audio to `{audio_path}`")
            print()

        counter += 1

    stats.save_results()

    synthesizer.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-speech streaming synthesis")

    parser.add_argument(
        "--dataset",
        default=TextDatasets.TASKMASTER2.value,
        choices=[d.value for d in TextDatasets],
        help="Choose type of input type")
    parser.add_argument(
        "--input-audio-device-index",
        type=int,
        default=-1,
        help="Index of input audio device")
    parser.add_argument(
        "--speech-endpoint-duration-sec",
        type=float,
        default=None,
        help="Duration in seconds for speechless audio to be considered an endpoint")
    parser.add_argument(
        "--show-audio-devices",
        action="store_true",
        help="Only list available devices and exit")

    parser.add_argument(
        "--llm",
        default=DEFAULT_LLM,
        choices=[llm.value for llm in LLMs],
        help="Choose LLM to use")
    parser.add_argument(
        "--openai-access-key",
        default=None,
        help="Open AI access key. Needed when using openai models")
    parser.add_argument(
        "--system-message",
        default=None,
        help="The system message to use to prompt the LLM response")

    parser.add_argument(
        "--tts",
        dest="synthesizer",
        default=Synthesizers.PICOVOICE_ORCA.value,
        choices=[s.value for s in Synthesizers],
        help="Choose voice synthesizer to use")
    parser.add_argument(
        "--picovoice-access-key",
        default=None,
        help="AccessKey obtained from Picovoice Console")
    parser.add_argument(
        "--orca-model-path",
        default=None,
        help="Path to the model parameters file")
    parser.add_argument(
        "--orca-library-path",
        default=None,
        help="Path to Orca's dynamic library")

    parser.add_argument(
        "--aws-profile-name",
        default=None,
        help="AWS profile name to use for AWS Polly")

    parser.add_argument(
        "--azure-speech-key",
        default=None,
        help="Azure access token")
    parser.add_argument(
        "--azure-speech-region",
        default=None,
        help="Azure speech location")

    parser.add_argument(
        "--ibm-watson-api-key",
        default=None,
        help="IBM Watson API key")
    parser.add_argument(
        "--ibm-watson-service-url",
        default=None,
        help="IBM Watson service URL")

    parser.add_argument(
        "--num-interactions",
        type=int,
        default=100,
        help="Number of interactions to benchmark")
    parser.add_argument(
        "--results-folder",
        default=DEFAULT_RESULTS_FOLDER,
        help="Folder to save results")

    main(parser.parse_args())
