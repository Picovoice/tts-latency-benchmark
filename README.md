# Text-to-Speech Latency Benchmark

Made in Vancouver, Canada by [Picovoice](https://picovoice.ai)

This repo is a minimalist and extensible framework for benchmarking the latency of different text-to-speech engines,
when used in conjunction with large language models (LLMs) for voice assistants.

## Table of Contents

- [Data](#data)
- [Metrics](#metrics)
- [Engines](#engines)
- [Usage](#usage)
- [Results](#results)

## Overview

- Picovoice's [Cheetah Streaming Speech-to-Text](https://picovoice.ai/platform/cheetah/)
- ChatGPT
- TTS

## Data

[taskmaster2](https://huggingface.co/datasets/taskmaster2)

## Metrics

Latency is typically measured with the `time-to-first-byte` (`TTFB`) metric, which is the time taken from the moment a
request was sent until the first byte is received.

In the context of voice assistants, the metric we care about is:

- **End-to-End Latency**: Time taken from the moment the user's request is sent to the LLM, until the TTS engine
  produces the first byte of audio.

The `end-to-end latency` in voice assistants is the sum of the `LLM TTFB` and `TTS TTFB`:

- **LLM Latency**: Time taken from the moment the user's request is sent to the LLM, until the LLM produces the first
  byte of text.
- **TTS Latency**: Time taken from the moment the LLM produces the first text chunk, until the TTS engine produces the
  first byte of audio.

Note that we omit the ASR latency for two main reasons:
First, it is very hard to measure accurately in a controlled environment and is highly context specific.
Second, since we use a real-time ASR engine
(Picovoice's [Cheetah Streaming Speech-to-Text](https://picovoice.ai/platform/cheetah/)) for all experiments,
we can assume that the ASR TTFB is constant across all experiments and only a small fraction of the total latency.

## TTS engines

All that have streaming audio capabilities:

- Amazon Polly
- Azure TTS
- ElevenLabs
- IBM Watson
- OpenAI TTS

We compare these to Picovoice Orca that does not only have output audio streaming capabilities, but also
allows for input text streaming.
