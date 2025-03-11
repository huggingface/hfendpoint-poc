from typing import Sequence

import numpy as np


def chunk_audio_with_duration(audio: np.ndarray, maximum_duration_sec: int, sampling_rate: int) -> Sequence[np.ndarray]:
    """
    Chunk a mono audio timeseries so that each chunk is as long as `maximum_duration_sec`.
    Chunks are evenly distributed except the last one which might be shorter
    :param audio: The mono timeseries waveform of the audio
    :param maximum_duration_sec: The maximum length, in seconds, for each chunk
    :param sampling_rate: The number of samples to represent one second of audio
    :return: List of numpy array representing the chunk
    """

    # We pad the input so that every chunk length is `max_duration_sec`
    max_duration_samples = sampling_rate * maximum_duration_sec
    padding = max_duration_samples - np.remainder(len(audio), max_duration_samples)
    audio = np.pad(audio, (0, padding), constant_values=0.0)
    return np.split(audio, len(audio) // max_duration_samples)