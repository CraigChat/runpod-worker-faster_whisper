import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence, Tuple

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import (
    Segment,
    TranscriptionOptions,
    Word,
    get_suppressed_tokens,
    restore_speech_timestamps,
)
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

from decode import SAMPLING_RATE, available_cores, decode_file

logger = logging.getLogger("runpod-worker")

CHUNK_LENGTH = 30
DEFAULT_BATCH_SIZE = 16

VAD_OPTIONS = VadOptions(
    max_speech_duration_s=CHUNK_LENGTH,
    min_silence_duration_ms=160,
    speech_pad_ms=200,
)

SILENCE_LEVEL = 1e-4
SILENCE_GUARD_S = 8.0
SILENCE_MIN_GAP_S = 18.0


def _batch_size() -> int:
    try:
        return max(1, int(os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE)))
    except ValueError:
        return DEFAULT_BATCH_SIZE


def _vad_regions(audio: np.ndarray) -> List[Tuple[int, int]]:
    """Sample ranges worth running Silero on: everything except long stretches of
    digital silence."""
    window = 512
    n = len(audio) // window
    if n == 0:
        return [(0, len(audio))]
    loud = np.abs(audio[: n * window]).reshape(n, window).max(axis=1) > SILENCE_LEVEL
    guard = int(SILENCE_GUARD_S * SAMPLING_RATE / window)
    loud = np.convolve(loud.astype(np.int32), np.ones(2 * guard + 1, np.int32), "same") > 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], loud.astype(np.int8), [0]))))

    regions: List[Tuple[int, int]] = []
    for start, end in zip(edges[::2] * window, edges[1::2] * window):
        end = min(len(audio), end)
        if regions and start - regions[-1][1] < SILENCE_MIN_GAP_S * SAMPLING_RATE:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))
    if regions and len(audio) - regions[-1][1] < window:
        regions[-1] = (regions[-1][0], len(audio))
    return regions


def _detect_speech(audio: np.ndarray, pool: ThreadPoolExecutor) -> List[dict]:
    def run(region):
        start, end = region
        return [
            {"start": c["start"] + start, "end": c["end"] + start}
            for c in get_speech_timestamps(audio[start:end], VAD_OPTIONS)
        ]

    return [clip for clips in pool.map(run, _vad_regions(audio)) for clip in clips]


def transcribe_tracks(
    model: WhisperModel,
    paths: Sequence[str],
    language: Optional[str] = None,
    beam_size: int = 5,
    patience: float = 1,
    length_penalty: float = 1,
    initial_prompt: Optional[str] = None,
    fallback_temperatures: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    log_prob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    word_timestamps: bool = True,
) -> List[Tuple[List[Segment], Optional[str]]]:
    """Transcribe audio files together. Returns (segments, language) per file, in order."""
    pipeline = BatchedInferencePipeline(model)
    batch_size = _batch_size()
    cores = available_cores()

    tracks: List[Optional[dict]] = [None] * len(paths)
    languages: List[Optional[str]] = [language] * len(paths)
    chunk_segments = {}  # (track index, chunk index) -> list of segment dicts
    contexts = {}  # language -> (tokenizer, options)

    def prepare(path: str) -> dict:
        audio = decode_file(path)
        clips = _detect_speech(audio, vad_pool)
        if not clips:
            return {"clips": [], "chunks": []}
        chunks, metadata = collect_chunks(audio, clips, max_duration=CHUNK_LENGTH)
        features = [model.feature_extractor(chunk)[..., :-1] for chunk in chunks]
        return {"clips": clips, "chunks": list(zip(chunks, features, metadata))}

    def detect_language(track: dict) -> str:
        features = [f for _, f, _ in track["chunks"]]
        lang, _, _ = model.detect_language(
            features=np.concatenate(
                features + [np.full((model.model.n_mels, 1), -1.5, dtype="float32")], axis=1
            )
        )
        return lang

    def context(lang: str):
        if lang not in contexts:
            tokenizer = Tokenizer(
                model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language=lang
            )
            contexts[lang] = (tokenizer, TranscriptionOptions(
                beam_size=beam_size,
                best_of=5,
                patience=patience,
                length_penalty=length_penalty,
                repetition_penalty=1,
                no_repeat_ngram_size=0,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
                condition_on_previous_text=False,
                prompt_reset_on_temperature=0.5,
                temperatures=[fallback_temperatures[0]],
                initial_prompt=initial_prompt,
                prefix=None,
                suppress_blank=True,
                suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
                without_timestamps=True,
                max_initial_timestamp=0.0,
                word_timestamps=word_timestamps,
                prepend_punctuations="\"'“¿([{-",
                append_punctuations="\"'.。,，!！?？:：”)]}、",
                multilingual=False,
                max_new_tokens=None,
                clip_timestamps=None,
                hallucination_silence_threshold=None,
                hotwords=None,
            ))
        return contexts[lang]

    def redecode(track_index: int, chunk_index: int, lang: str) -> List[dict]:
        """Sequential decode of one chunk with the full temperature-fallback schedule."""
        audio, _, metadata = tracks[track_index]["chunks"][chunk_index]
        offset = metadata["offset"]
        segments, _ = model.transcribe(
            audio,
            language=lang,
            beam_size=beam_size,
            patience=patience,
            length_penalty=length_penalty,
            temperature=list(fallback_temperatures),
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            vad_filter=False,
        )
        return [
            dict(
                text=s.text, tokens=s.tokens, start=s.start + offset, end=s.end + offset,
                avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
                compression_ratio=s.compression_ratio, seek=s.seek,
                words=[
                    dict(word=w.word, start=w.start + offset, end=w.end + offset,
                         probability=w.probability)
                    for w in (s.words or [])
                ],
            )
            for s in segments
        ]

    def run_batch(batch, lang: str):
        tokenizer, options = context(lang)
        features = np.stack([pad_or_trim(f) for _, _, f, _ in batch])
        pipeline.last_speech_timestamp = 0.0
        outputs = pipeline.forward(features, tokenizer, [m for *_, m in batch], options)
        for (track_index, chunk_index, _, _), segments in zip(batch, outputs):
            if (
                segments
                and no_speech_threshold is not None
                and segments[0]["no_speech_prob"] > no_speech_threshold
                and log_prob_threshold is not None
                and segments[0]["avg_logprob"] < log_prob_threshold
            ):
                segments = []
            if compression_ratio_threshold is not None and any(
                s["compression_ratio"] > compression_ratio_threshold for s in segments
            ):
                logger.info("Re-decoding chunk %d of track %d with fallback", chunk_index, track_index)
                segments = redecode(track_index, chunk_index, lang)
            chunk_segments[(track_index, chunk_index)] = segments

    pending = defaultdict(list)  # language -> [(track, chunk, features, metadata)]
    with ThreadPoolExecutor(cores) as vad_pool, \
            ThreadPoolExecutor(min(len(paths), cores)) as prep_pool:
        futures = {prep_pool.submit(prepare, path): i for i, path in enumerate(paths)}
        for future in as_completed(futures):
            track_index = futures[future]
            track = tracks[track_index] = future.result()
            if not track["chunks"]:
                continue
            lang = languages[track_index] = language or detect_language(track)
            queue = pending[lang]
            queue.extend(
                (track_index, chunk_index, features, metadata)
                for chunk_index, (_, features, metadata) in enumerate(track["chunks"])
            )
            # keep the GPU busy while the remaining tracks are still being prepared
            while len(queue) >= batch_size:
                run_batch(queue[:batch_size], lang)
                del queue[:batch_size]

    for lang, queue in pending.items():
        for i in range(0, len(queue), batch_size):
            run_batch(queue[i:i + batch_size], lang)

    results = []
    for track_index, track in enumerate(tracks):
        segments: List[Segment] = []
        for chunk_index in range(len(track["chunks"])):
            for s in chunk_segments.get((track_index, chunk_index), []):
                segments.append(Segment(
                    id=len(segments) + 1,
                    seek=s["seek"],
                    start=round(s["start"], 3),
                    end=round(s["end"], 3),
                    text=s["text"],
                    tokens=s["tokens"],
                    avg_logprob=s["avg_logprob"],
                    compression_ratio=s["compression_ratio"],
                    no_speech_prob=s["no_speech_prob"],
                    words=[Word(**w) for w in s.get("words", [])] if word_timestamps else None,
                    temperature=fallback_temperatures[0],
                ))
        if track["clips"]:
            segments = list(restore_speech_timestamps(segments, track["clips"], SAMPLING_RATE))
        results.append((segments, languages[track_index]))
    return results
