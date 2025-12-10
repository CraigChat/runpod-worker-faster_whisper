"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import asyncio
import base64
import logging
import os
import tempfile
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, List, cast
import warnings

from deepmultilingualpunctuation import PunctuationModel
from corrector import Word, add_ellipsis_to_words, correct_words
from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict

logger = logging.getLogger("runpod-worker")

MODEL = predict.Predictor()
MODEL.setup()
PUNCTUATION_MODEL = PunctuationModel("kredor/punctuate-all")


@contextmanager
def timed_step(name: str, timings: Dict[str, float]):
    start = perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (perf_counter() - start)


def resolve_max_concurrency(total_items: int) -> int:
    if total_items <= 1:
        return 1

    env_value = os.getenv("MAX_CONCURRENCY")
    if env_value is not None:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return min(parsed, total_items)
        except ValueError:
            logger.warning(
                "Invalid MAX_CONCURRENCY value '%s'; defaulting to %d.",
                env_value,
                total_items,
            )

    return total_items


def log_request_completion(
    job_id: str,
    status: str,
    timings: Dict[str, float],
    audio_count: int,
    max_concurrency: int,
    correction_enabled: bool,
):
    logger.info(
        "Job %s status=%s audio_count=%d max_concurrency=%d correction_enabled=%s "
        "total=%.3fs validation=%.3fs input_preparation=%.3fs transcription=%.3fs "
        "correction=%.3fs cleanup=%.3fs",
        job_id,
        status,
        audio_count,
        max_concurrency,
        correction_enabled,
        timings.get("total", 0.0),
        timings.get("validation", 0.0),
        timings.get("input_preparation", 0.0),
        timings.get("transcription", 0.0),
        timings.get("correction", 0.0),
        timings.get("cleanup", 0.0),
    )


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction
    '''
    job_input_raw = cast(Dict[str, Any], job.get('input', {}))
    job_input: Dict[str, Any] = job_input_raw

    timings: Dict[str, float] = {}
    overall_start = perf_counter()
    status = "completed"
    audio_inputs: List[str] = []
    applied_concurrency = 1
    correction_enabled = bool(job_input.get("transcription_corrector", False))

    try:
        with timed_step("validation", timings), rp_debugger.LineTimer('validation_step'):
            input_validation = validate(job_input, INPUT_VALIDATIONS)

            if 'errors' in input_validation:
                status = "validation_error"
                return {"error": input_validation['errors']}

        job_input = cast(Dict[str, Any], input_validation['validated_input'])
        correction_enabled = job_input["transcription_corrector"]

        audio_fields = [
            bool(job_input.get('audio', False)),
            bool(job_input.get('audio_base64', False)),
            bool(job_input.get('audios', False))
        ]
        field_count = sum(audio_fields)
        if field_count == 0:
            status = "missing_audio"
            return {'error': 'Must provide one of audio, audio_base64, or audios'}
        if field_count > 1:
            status = "multiple_audio_fields"
            return {'error': 'Must provide only one of audio, audio_base64, or audios'}

        with timed_step("input_preparation", timings):
            if job_input.get('audio', False):
                with rp_debugger.LineTimer('download_step'):
                    audio_inputs = [download_files_from_urls(job['id'], [job_input['audio']])[0]]
            elif job_input.get('audio_base64', False):
                audio_inputs = [base64_to_tempfile(job_input['audio_base64'])]
            elif job_input.get('audios', False):
                audios = job_input['audios']
                if not isinstance(audios, list) or len(audios) == 0:
                    status = "invalid_audio_list"
                    return {'error': 'audios must be a non-empty list'}
                if isinstance(audios[0], str) and audios[0].startswith('http'):
                    with rp_debugger.LineTimer('download_step'):
                        audio_inputs = download_files_from_urls(job['id'], audios)
                else:
                    audio_inputs = [base64_to_tempfile(b64) for b64 in audios]

        logger.info(f"Number of audio inputs: {len(audio_inputs)}")

        applied_concurrency = resolve_max_concurrency(len(audio_inputs))

        predict_kwargs = dict(
            model_name=job_input["model"],
            transcription=job_input["transcription"],
            translation=job_input["translation"],
            translate=job_input["translate"],
            language=job_input["language"],
            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],
            temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            enable_vad=job_input["enable_vad"],
            word_timestamps=job_input["word_timestamps"]
        )

        def run_prediction(audio_path: str):
            return MODEL.predict(audio=audio_path, **predict_kwargs)

        def sequential_predict():
            return [run_prediction(path) for path in audio_inputs]

        with timed_step("transcription", timings), rp_debugger.LineTimer('prediction_step'):
            if len(audio_inputs) <= 1 or applied_concurrency <= 1:
                results = sequential_predict()
            else:
                async def transcribe_all():
                    semaphore = asyncio.Semaphore(applied_concurrency)

                    async def predict_with_limit(path: str):
                        async with semaphore:
                            return await asyncio.to_thread(run_prediction, path)

                    coroutines = [predict_with_limit(path) for path in audio_inputs]
                    return await asyncio.gather(*coroutines)

                try:
                    results = asyncio.run(transcribe_all())
                except RuntimeError:
                    results = sequential_predict()
                    applied_concurrency = 1

        with timed_step("cleanup", timings), rp_debugger.LineTimer('cleanup_step'):
            rp_cleanup.clean(['input_objects'])

        if not correction_enabled:
            status = "completed"
            return results[0] if len(results) == 1 else results

        all_words: List[Word] = []
        with timed_step("correction", timings), rp_debugger.LineTimer('punctuation_correction_step'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for track_index, result in enumerate(results):
                    track_words: List[Word] = []
                    if "segments" in result:
                        for segment in result["segments"]:
                            if "words" in segment:
                                for word in segment["words"]:
                                    if word["word"]:
                                        track_words.append(Word(
                                            track=track_index,
                                            word=word["word"],
                                            start=word["start"],
                                            end=word["end"]
                                        ))

                    if len(track_words) > 1:
                        add_ellipsis_to_words(track_words)
                        all_words.extend(track_words)

                    # with open(f"jobs/track_{track_index}.words.json", 'w') as f:
                    #     f.write(json.dumps(track_words, cls=WordEncoder, indent=2))

            (corrected_lines, change_count) = correct_words(all_words, PUNCTUATION_MODEL, logger)

            # with open(f"jobs/normalized_words.json", 'w') as f:
            #     lines = [f"Speaker {l.track + 1 if l.track is not None else 'NULL'}: \"{l.text}\"" for l in corrected_lines]
            #     f.write(json.dumps({
            #         "transcription": "\n".join(lines),
            #         "change_count": change_count,
            #         "corrected_lines": corrected_lines,
            #         "results": results
            #     }, cls=WordEncoder, indent=2))

        status = "completed"
        return {
            "corrected_segments": corrected_lines,
            "change_count": change_count,
            "results": results
        }
    except Exception:
        status = "exception"
        raise
    finally:
        total_elapsed = perf_counter() - overall_start
        timings["total"] = total_elapsed
        log_request_completion(
            job.get("id", "unknown"),
            status,
            timings,
            len(audio_inputs),
            applied_concurrency,
            correction_enabled,
        )

runpod.serverless.start({"handler": run_whisper_job})