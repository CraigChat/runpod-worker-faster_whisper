# Faster Whisper Runpod Worker

This is a modified version of [Runpod's Faster Whisper worker](https://github.com/runpod-workers/worker-faster_whisper) to change the format to return only Craig needs, put word timestamps in segments, and allow for multiple audio inputs.

This also incorporates a modified method of assembling (or correcting) multiple transcriptions into a single list of lines from [Kadda OK's TASMAS](https://github.com/KaddaOK/TASMAS). This uses [Deep Multilingual Punctuation Prediction](https://github.com/oliverguhr/deepmultilingualpunctuation/) using the [kredor/punctuate-all](https://huggingface.co/kredor/punctuate-all) model for correcting punctuations.