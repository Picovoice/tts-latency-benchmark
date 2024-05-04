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
    Synthesizers.OPENAI: 'OpenAI TTS',
    Synthesizers.PICOVOICE_ORCA: 'Picovoice\nOrca',
}

ENGINE_COLORS = {
    Synthesizers.OPENAI: GREY1,
    Synthesizers.PICOVOICE_ORCA: BLUE,
}


def _plot_time_first_audio(
        save_folder: str,
        show: bool = False,
        show_error_bars: bool = True,
) -> None:
    results = []
    for file in os.listdir(save_folder):
        if file.endswith(".json"):
            json_path = os.path.join(save_folder, file)
            synthesizer, mean, std = Stats.load_results(json_path)
            results.append((synthesizer, mean, std))

    results = sorted(results, key=lambda x: x[1].num_seconds_total_delay, reverse=True)

    print("RESULTS\n")
    max_delay = 0
    for synthesizer, mean, std in results:
        print(f"TTS: {synthesizer.value}")
        print(f"Total delay: {mean.num_seconds_total_delay:.2f} +- {std.num_seconds_total_delay:.2f} seconds")
        print(f"LLM delay: {mean.num_seconds_first_token:.2f} +- {std.num_seconds_first_token:.2f} seconds")
        print(f"TTS delay: {mean.num_seconds_first_audio:.2f} +- {std.num_seconds_first_audio:.2f} seconds\n")
        max_delay = max(max_delay, mean.num_seconds_total_delay)

    fig, ax = plt.subplots(figsize=(12, 6))

    # for i, (synthesizer, mean, std) in enumerate(results):
    #     rounded_result = round(1000 * mean.num_seconds_total_delay, -1)
    #     color = ENGINE_COLORS[synthesizer]
    #     ax.bar([i], [rounded_result], 0.4, color=color)
    #     ax.text(i - 0.04, rounded_result + 100, f'{rounded_result:.0f} ms', color=color, fontsize=12)

    def round_result(value: float) -> float:
        return round(1000 * value, -1)

    rounded_results = []
    colors = []
    bottoms = []
    for synthesizer, mean, std in results:
        rounded_result = round_result(mean.num_seconds_first_token)
        rounded_results.append(rounded_result)
        colors.append(ENGINE_COLORS[synthesizer])
        bottoms.append(rounded_result)
    ax.bar(range(len(results)), rounded_results, 0.4, color=colors, label="LLM delay")

    rounded_results = []
    colors = []
    for i, (synthesizer, mean, std) in enumerate(results):
        rounded_results.append(round_result(mean.num_seconds_first_audio))
        colors.append(ENGINE_COLORS[synthesizer])
    ax.bar(range(len(results)), rounded_results, 0.4, color=colors, bottom=bottoms, alpha=0.6, label="TTS delay")

    total_delays = []
    total_delays_std = []
    for i, (synthesizer, mean, std) in enumerate(results):
        rounded_result = round_result(mean.num_seconds_total_delay)
        total_delays.append(rounded_result)
        total_delays_std.append(round_result(std.num_seconds_total_delay))
        color = ENGINE_COLORS[synthesizer]
        x_offset = 0.08 if show_error_bars else -0.04
        ax.text(i + x_offset, rounded_result + 100, f'{rounded_result:.0f} ms', color=color, fontsize=12)

    if show_error_bars:
        plt.errorbar(
            range(len(results)),
            total_delays,
            total_delays_std,
            fmt='.',
            color='Black',
            alpha=0.5,
        )

    for spine in plt.gca().spines.values():
        if spine.spine_type != 'bottom' and spine.spine_type != 'left':
            spine.set_visible(False)

    plt.xticks(np.arange(0, len(Synthesizers)), [ENGINE_PRINT_NAMES[x[0]] for x in results], fontsize=12)
    y_arange = np.arange(0, 1000 * (max_delay + (max_delay / 5)), 500)
    plt.yticks(y_arange, [f"{x:.0f}" for x in y_arange])
    plt.ylabel('Time to first audio (ms)', fontsize=14)

    ax.legend(loc="upper right", fontsize=14, reverse=True)

    plot_path = os.path.join(save_folder, "time_to_first_audio.png")
    if show_error_bars:
        plot_path = plot_path.replace(".png", "_error_bars.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Saved plot to `{plot_path}`")

    if show:
        plt.show()

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", default=DEFAULT_RESULTS_FOLDER, help="Path to results folder")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    save_folder = os.path.join(args.results_folder, f"llm_{DEFAULT_LLM}")

    _plot_time_first_audio(save_folder=save_folder, show=args.show, show_error_bars=args.show_errors)


if __name__ == "__main__":
    main()
