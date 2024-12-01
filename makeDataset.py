import os
import tensorflow as tf
from preprocess import parse_and_process_file


# 데이터셋 경로 설정
base_path = "/Users/songdohyeon/Downloads/dataverse_files"
measurement_series = [
    "measurementSeries_B",
    "measurementSeries_C",
    "measurementSeries_D",
    "measurementSeries_E",
    "measurementSeries_F"
]
tightening_levels = ["05cNm", "10cNm", "20cNm", "30cNm", "40cNm", "50cNm", "60cNm"]

# 라벨 매핑
label_mapping = {
    "05cNm": 0,
    "10cNm": 0,
    "20cNm": 0,
    "30cNm": 0,
    "40cNm": 0,
    "50cNm": 1,
    "60cNm": 1
}

# 파일 경로와 라벨 리스트 생성
file_paths = []
labels = []

for series in measurement_series:
    for level in tightening_levels:
        folder_path = os.path.join(base_path, series, level)
        if not os.path.exists(folder_path):
            continue
        mat_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mat")]
        file_paths.extend(mat_files)
        labels.extend([label_mapping[level]] * len(mat_files))

# TensorFlow 데이터셋으로 변환
file_paths = tf.constant(file_paths)
labels = tf.constant(labels, dtype=tf.int32)
batch_size = 16
nperseg = 1024
noverlap = 768
original_shape = (512,256)
downsampled_shape = (256, 128)  # Downsampled shape
buffer_size = 100
num_classes = 2
def createTfDataset():
    # TensorFlow 데이터셋 생성
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(
            lambda file_path, label: (
                parse_and_process_file(file_path, nperseg, noverlap, original_shape),
                tf.one_hot(label, depth=num_classes)  # 레이블을 원-핫 인코딩
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.map(
            lambda x, y: (tf.image.resize(x, downsampled_shape), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Prefetch to improve performance
    
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)  # drop_remainder 적용
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset