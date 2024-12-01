from scipy.signal import stft
import numpy as np
import tensorflow as tf
import scipy.io

def apply_stft(data, nperseg=1024, noverlap=768, fixed_shape=(512, 256)):
    """
    Applies STFT to the given signal and returns a fixed-shape spectrogram.
    """
    _, _, Zxx = stft(data, nperseg=nperseg, noverlap=noverlap)
    spectrogram = np.abs(Zxx)

    # Pad or crop to fixed shape
    padded_spectrogram = np.zeros(fixed_shape, dtype=np.float32)
    min_freq, min_time = min(spectrogram.shape[0], fixed_shape[0]), min(spectrogram.shape[1], fixed_shape[1])
    padded_spectrogram[:min_freq, :min_time] = spectrogram[:min_freq, :min_time]
    return padded_spectrogram


def parse_and_process_file(file_path, nperseg=1024, noverlap=768, original_shape=(512, 256)):
    def load_and_process(filepath):
        filepath_str = filepath.numpy().decode("utf-8")  # TensorFlow 텐서를 Python 문자열로 변환
        mat_data = scipy.io.loadmat(filepath_str)

        # Extract signals A, B, C
        data_A = mat_data.get('A', np.zeros((1024, 1))).flatten()
        data_B = mat_data.get('B', np.zeros((1024, 1))).flatten()
        data_C = mat_data.get('C', np.zeros((1024, 1))).flatten()

        # Apply STFT
        spectrogram_A = apply_stft(data_A, nperseg, noverlap, original_shape)
        spectrogram_B = apply_stft(data_B, nperseg, noverlap, original_shape)
        spectrogram_C = apply_stft(data_C, nperseg, noverlap, original_shape)

        # Combine spectrograms
        combined_spectrogram = np.stack([spectrogram_A, spectrogram_B, spectrogram_C], axis=-1)

        # Normalize spectrogram
        combined_spectrogram = (combined_spectrogram - np.min(combined_spectrogram)) / (
            np.max(combined_spectrogram) - np.min(combined_spectrogram) + 1e-8
        )
        return combined_spectrogram

    spectrogram = tf.py_function(
        func=load_and_process,
        inp=[file_path],
        Tout=tf.float32
    )
    spectrogram.set_shape(original_shape + (3,))
    return spectrogram

def create_optimized_dataset(file_paths, labels, batch_size=16, nperseg=1024, noverlap=768, fixed_shape=(512, 256), num_classes=2):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(
        lambda file_path, label: (
            parse_and_process_file(file_path, nperseg, noverlap, fixed_shape),
            tf.cast(tf.one_hot(label, depth=num_classes), tf.float32)  # 레이블을 원-핫 인코딩
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
