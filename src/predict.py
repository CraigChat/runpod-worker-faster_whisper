"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

import gc
import os
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
)  # Still needed for transcribe potentially?
import numpy as np

from runpod.serverless.utils import rp_cuda

from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp
from faster_whisper.transcribe import Segment

import batched

# rp_cuda.is_available() shells out to nvidia-smi; check once instead of on every call.
CUDA_AVAILABLE = rp_cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
COMPUTE_TYPE = "float16" if CUDA_AVAILABLE else "int8"

# Define available models (for validation)
AVAILABLE_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "distil-large-v2",
    "distil-large-v3",
    "turbo",
    "distil-large-v3.5",
}


class Predictor:
    """A Predictor class for the Whisper model with lazy loading"""

    def __init__(self):
        """Initializes the predictor with no models loaded."""
        self.models = {}
        self.model_lock = (
            threading.Lock()
        )  # Lock for thread-safe model loading/unloading

    def setup(self):
        """Pre-load PRELOAD_MODEL (turbo by default) so the first job on a cold worker
        doesn't pay for loading it. Set PRELOAD_MODEL to an empty string to disable."""
        model_name = os.getenv("PRELOAD_MODEL", "turbo")
        if model_name:
            self.get_model(model_name)

    def get_model(self, model_name):
        """Return the requested model, loading it (and unloading any other) if needed."""
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models are: {AVAILABLE_MODELS}"
            )

        with self.model_lock:
            if model_name in self.models:
                print(f"Using already loaded model: {model_name}")
                return self.models[model_name]

            # Unload existing model if necessary
            if self.models:
                existing_model_name = list(self.models.keys())[0]
                print(f"Unloading model: {existing_model_name}...")
                self.models.clear()
                # Hint Python to release memory
                gc.collect()
                print(f"Model {existing_model_name} unloaded.")

            print(f"Loading model: {model_name}...")
            try:
                model = WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                raise ValueError(f"Failed to load model {model_name}: {e}") from e
            self.models[model_name] = model
            print(f"Model {model_name} loaded successfully.")
            return model

    def predict_batched(
        self,
        audios,
        model_name="base",
        transcription="none",
        language=None,
        beam_size=5,
        patience=1,
        length_penalty=None,
        initial_prompt=None,
        temperature=0,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=True,
    ):
        """
        Transcribe several tracks together; see batched.py. Returns one result per
        track, in the same shape as predict().
        """
        model = self.get_model(model_name)

        if temperature_increment_on_fallback is not None:
            temperatures = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperatures = (temperature,)

        tracks = batched.transcribe_tracks(
            model,
            audios,
            language=language,
            beam_size=beam_size,
            patience=patience,
            # rp_schema defaults this to 0; the batched path was validated with
            # faster-whisper's default of 1 (normalise scores by length)
            length_penalty=length_penalty or 1,
            initial_prompt=initial_prompt,
            fallback_temperatures=temperatures,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            word_timestamps=word_timestamps,
        )

        return [
            {
                "segments": serialize_segments(segments),
                "detected_language": detected_language,
                "transcription": format_segments(transcription, segments),
                "translation": None,
                "device": DEVICE,
                "model": model_name,
            }
            for segments, detected_language in tracks
        ]

    def predict(
        self,
        audio,
        model_name="base",
        transcription="none",
        translate=False,
        translation="plain_text",  # Added in a previous PR
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=True,
        word_timestamps=True,
    ):
        """
        Run a single prediction on the model, loading/unloading models as needed.
        `audio` is a file path or an already-decoded 16 kHz mono float32 array.
        """
        if not isinstance(audio, np.ndarray):
            audio = str(audio)
        model = self.get_model(model_name)

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        # Note: FasterWhisper's transcribe might release the GIL, potentially allowing
        # other threads to acquire the model_lock if transcribe is lengthy.
        # If issues arise, the lock might need to encompass the transcribe call too.
        results = model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=[-1],  # Might need conversion from string
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=word_timestamps,
            vad_filter=enable_vad,
            vad_parameters={
                "min_speech_duration_ms": 0.04
            }
        )

        info = results[1]
        segments: list[Segment] = list(results[0])

        # Format transcription
        transcription_output = format_segments(transcription, segments)

        # Handle translation if requested
        translation_output = None
        if translate:
            translation_segments, _ = model.transcribe(
                audio,
                task="translate",
                temperature=temperature,  # Reuse temperature settings for translation
            )
            translation_output = format_segments(
                translation, list(translation_segments)
            )

        results = {
            "segments": serialize_segments(segments),
            "detected_language": info.language,
            "transcription": transcription_output,
            "translation": translation_output,
            "device": DEVICE,
            "model": model_name,
        }

        return results


def serialize_segments(transcript: list[Segment]):
    """
    Serialize the segments to be returned in the API response.
    """
    return [
        {
            "id": segment.id,
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                }
                for word in (segment.words or [])
            ],
        }
        for segment in transcript
    ]


def format_segments(format_type, segments):
    """
    Format the segments to the desired format
    """

    if format_type == "plain_text":
        return " ".join([segment.text.lstrip() for segment in segments])
    elif format_type == "formatted_text":
        return "\n".join([segment.text.lstrip() for segment in segments])
    elif format_type == "srt":
        return write_srt(segments)
    elif format_type == "vtt":  # Added VTT case
        return write_vtt(segments)
    elif format_type == "none":
        return None
    else:  # Default or unknown format
        print(f"Warning: Unknown format '{format_type}', defaulting to plain text.")
        return " ".join([segment.text.lstrip() for segment in segments])


def write_vtt(transcript):
    """
    Write the transcript in VTT format.
    """
    result = ""

    for segment in transcript:
        # Using the consistent timestamp format from previous PR
        result += f"{format_timestamp(segment.start, always_include_hours=True)} --> {format_timestamp(segment.end, always_include_hours=True)}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result


def write_srt(transcript):
    """
    Write the transcript in SRT format.
    """
    result = ""

    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result
