import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from benchmark import (
    DEFAULT_LLM,
    DEFAULT_RESULTS_FOLDER,
    Stats,
)
from tts import Synthesizers

Color = Tuple[float, float, float]


def rgb_from_hex(x: str) -> Color:
    x = x.strip("# ")
    assert len(x) == 6
    return int(x[:2], 16) / 255, int(x[2:4], 16) / 255, int(x[4:], 16) / 255


BLACK = rgb_from_hex("#000000")
GREY1 = rgb_from_hex("#3F3F3F")
GREY2 = rgb_from_hex("#5F5F5F")
GREY3 = rgb_from_hex("#7F7F7F")
GREY4 = rgb_from_hex("#9F9F9F")
GREY5 = rgb_from_hex("#BFBFBF")
WHITE = rgb_from_hex("#FFFFFF")
BLUE = rgb_from_hex("#377DFF")

ENGINE_PRINT_NAMES = {
    Synthesizers.AMAZON_POLLY: 'Amazon Polly',
    Synthesizers.AZURE_TTS: 'Azure TTS',
    Synthesizers.OPENAI_TTS: 'OpenAI TTS',
    Synthesizers.ELEVENLABS: 'ElevenLabs',
    Synthesizers.IBM_WATSON_TTS: 'IBM Watson\nTTS',
    Synthesizers.PICOVOICE_ORCA: 'Picovoice\nOrca',
}

ENGINE_COLORS = {
    Synthesizers.AMAZON_POLLY: GREY1,
    Synthesizers.AZURE_TTS: GREY2,
    Synthesizers.ELEVENLABS: GREY3,
    Synthesizers.IBM_WATSON_TTS: GREY3,
    Synthesizers.OPENAI_TTS: GREY3,
    Synthesizers.PICOVOICE_ORCA: BLUE,
}

ORDER = [
    Synthesizers.AMAZON_POLLY,
    Synthesizers.AZURE_TTS,
    Synthesizers.ELEVENLABS,
    Synthesizers.IBM_WATSON_TTS,
    Synthesizers.OPENAI_TTS,
    Synthesizers.PICOVOICE_ORCA]


def _plot_time_first_audio(
        save_folder: str,
        show: bool = False,
        show_error_bars: bool = True,
        only_tts: bool = False,
) -> None:
    raw_results = []
    for file in os.listdir(save_folder):
        if file.endswith(".json"):
            json_path = os.path.join(save_folder, file)
            synthesizer, mean, std = Stats.load_results(json_path, scale=1000)
            raw_results.append((synthesizer, mean, std))
    # filter out ibm watson
    raw_results = [x for x in raw_results if x[0] is not Synthesizers.IBM_WATSON_TTS]

    results = []
    for synthesizer in ORDER:
        for raw_result in raw_results:
            if raw_result[0] is synthesizer:
                results.append(raw_result)
                break

    num_results = len(results)

    print("RESULTS\n")
    max_delay = 0
    for synthesizer, mean, std in results:
        print(
            f"TTS: {synthesizer.value}")
        print(
            f"Total delay: {mean.total_delay_seconds:.2f} +- {std.total_delay_seconds:.2f} seconds")
        print(
            f"Delay caused by LLM: {mean.first_token_delay_seconds:.2f} +- {std.first_token_delay_seconds:.2f} seconds")
        print(
            f"Delay caused by TTS: {mean.first_audio_delay_seconds:.2f} +- "
            f"{std.first_audio_delay_seconds:.2f} seconds\n")
        max_delay = max(max_delay, mean.total_delay_seconds)

    fig, ax = plt.subplots(figsize=(12, 6))

    def round_result(value: float) -> float:
        return round(value, -1)

    if not only_tts:
        rounded_results = []
        colors = []
        bottoms = []
        for synthesizer, mean, std in results:
            rounded_result = round_result(mean.first_token_delay_seconds)
            rounded_results.append(rounded_result)
            colors.append(ENGINE_COLORS[synthesizer])
            bottoms.append(rounded_result)
        ax.bar(
            range(num_results),
            rounded_results,
            0.4,
            color=colors,
            label="Delay caused by LLM")
    else:
        bottoms = [0 for _ in range(num_results)]

    rounded_results = []
    colors = []
    for i, (synthesizer, mean, std) in enumerate(results):
        rounded_results.append(round_result(mean.first_audio_delay_seconds))
        colors.append(ENGINE_COLORS[synthesizer])
    ax.bar(
        range(num_results),
        rounded_results,
        0.4,
        color=colors,
        bottom=bottoms,
        alpha=0.65 if not only_tts else 1.0,
        label="Delay caused by TTS")

    total_delays = []
    total_delays_std = []
    for i, (synthesizer, mean, std) in enumerate(results):
        mean_total_delay = mean.total_delay_seconds if not only_tts else mean.first_audio_delay_seconds
        rounded_result = round_result(mean_total_delay)
        total_delays.append(rounded_result)
        std_total_delay = std.total_delay_seconds if not only_tts else std.first_audio_delay_seconds
        total_delays_std.append(round_result(std_total_delay))
        color = ENGINE_COLORS[synthesizer]
        x_offset = 0.08 if show_error_bars else -0.2
        ax.text(
            i + x_offset, rounded_result + 70,
            f'{rounded_result:.0f} ms',
            color=color,
            fontsize=12)

    if show_error_bars:
        plt.errorbar(
            range(num_results),
            total_delays,
            total_delays_std,
            fmt='.',
            color='Black',
            alpha=0.5,
        )

    for spine in plt.gca().spines.values():
        if spine.spine_type != 'bottom' and spine.spine_type != 'left':
            spine.set_visible(False)

    plt.xticks(np.arange(0, len(rounded_results)), [ENGINE_PRINT_NAMES[x[0]] for x in results], fontsize=12)
    y_arange = np.arange(0, (max_delay + (max_delay / 5)), 500)
    plt.yticks(y_arange, [f"{x:.0f}" for x in y_arange])
    metric = "End-to-End Latency" if not only_tts else "Text-to-Speech Latency"
    plt.ylabel(f"Average {metric} (ms)", fontsize=14)

    if not only_tts:
        ax.legend(loc="upper left", reverse=True, fontsize=14)

    plot_path = os.path.join(save_folder, "time_to_first_audio.png")
    if show_error_bars:
        plot_path = plot_path.replace(".png", "_error_bars.png")
    if only_tts:
        plot_path = plot_path.replace(".png", "_only_tts.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Saved plot to `{plot_path}`")

    if show:
        plt.show()

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder",
        default=DEFAULT_RESULTS_FOLDER,
        help="Path to results folder")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--only-tts", action="store_true")
    args = parser.parse_args()

    save_folder = os.path.join(args.results_folder, f"llm_{DEFAULT_LLM}")

    _plot_time_first_audio(
        save_folder=save_folder,
        show=args.show,
        show_error_bars=args.show_errors,
        only_tts=args.only_tts,
    )


if __name__ == "__main__":
    main()
