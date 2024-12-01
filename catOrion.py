import tensorflow as tf
import scipy.io
import numpy as np
import os
from scipy.signal import stft

import os

import numpy as np
import pandas as pd

import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Sequential, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from collections import Counter

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from preprocess import create_optimized_dataset

# Step 1: Set up dataset paths and labels
base_path = "/Users/songdohyeon/Downloads/dataverse_files"
measurement_series = [
    "measurementSeries_B",
    "measurementSeries_C",
    "measurementSeries_D",
    "measurementSeries_E",
    "measurementSeries_F"
]
tightening_levels = ["05cNm", "10cNm", "20cNm", "30cNm", "40cNm", "50cNm", "60cNm"]

file_paths = []
labels = []

for series in measurement_series:
    for level in tightening_levels:
        folder_path = os.path.join(base_path, series, level)
        if not os.path.exists(folder_path):
            continue
        mat_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mat")]
        file_paths.extend(mat_files)
        labels.extend([1 if level in ["05cNm", "10cNm", "20cNm", "30cNm", "40cNm"] else 0] * len(mat_files))

file_paths = tf.constant(file_paths)
labels = tf.constant(labels)


# Step 2: Create TensorFlow dataset with downsampling
batch_size = 16
nperseg = 1024
noverlap = 768
original_shape = (512,256)
downsampled_shape = (256, 128)  # Downsampled shape

def create_downsampled_dataset(file_paths, labels, batch_size, nperseg, noverlap, original_shape, downsampled_shape):
    """
    Creates a TensorFlow dataset with downsampled spectrogram images.
    """
    # Create the optimized dataset
    dataset = create_optimized_dataset(file_paths, labels, batch_size, nperseg, noverlap, original_shape)

    # Apply resizing to downsample spectrogram images
    dataset = dataset.map(
        lambda x, y: (tf.image.resize(x, downsampled_shape), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch to improve performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create the downsampled TensorFlow dataset
tf_dataset = create_downsampled_dataset(
    file_paths, labels, batch_size, nperseg, noverlap, original_shape, downsampled_shape
)


# Step 3: Verify dataset and statistics
print("Data Load Summary:")
print(f"  Total files: {len(file_paths)}")
print(f"  Label distribution: {dict(zip(*np.unique(labels.numpy(), return_counts=True)))}")

for batch_data, batch_labels in tf_dataset.take(1):
    print(f"Batch data shape: {batch_data.shape}")
    print(f"Batch labels: {batch_labels.numpy()}")
    
    from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

def build_generator(latent_dim, num_classes, target_shape=(256, 128, 3)):
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(num_classes,))
    combined_input = tf.keras.layers.Concatenate()([noise, label])

    # Dense layer output matches starting Reshape size
    x = Dense(32 * 16 * 128, activation='relu')(combined_input)  # Match 32x16x128
    x = Reshape((32, 16, 128))(x)  # Start with 32x16x128

    # Upsample to target shape
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # -> 64x32x128
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)   # -> 128x64x64
    x = BatchNormalization()(x)
    x = Conv2DTranspose(target_shape[-1], (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x)  # -> 256x128x3

    generator = Model([noise, label], x, name="Generator")
    return generator

latent_dim = 100
num_classes = 2
generator = build_generator(latent_dim, num_classes)

# Generate a sample image
noise = tf.random.normal([1, latent_dim])
label = tf.one_hot([1], num_classes)
generated_image = generator([noise, label])

print(f"Generated image shape: {generated_image.shape}")
# Expected Output: (1, 256, 128, 3)

from tensorflow.keras.layers import Conv2D, Flatten, Dropout, GlobalAveragePooling2D

def build_discriminator(input_shape=(256, 128, 3), num_classes=2):
    inputs = Input(shape=input_shape)

    # First Conv Block
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)ㅅ

    # Second Conv Block
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Outputs
    validity = Dense(1, activation='sigmoid', name="Validity")(x)
    label = Dense(num_classes, activation='softmax', name="Class")(x)

    discriminator = Model(inputs, [validity, label], name="Discriminator")
    return discriminator


import tensorflow as tf

latent_dim = 100
num_classes = 2
generator = build_generator(latent_dim, num_classes)
discriminator = build_discriminator()
epochs = 50
lambda_coeff = 1.0


# Optimizers
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# 손실 함수 정의
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
scce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# 학습 단계 함수 정의
def train_step(real_images, real_labels):
    batch_size = tf.shape(real_images)[0]

    # 노이즈 및 랜덤 라벨 생성
    noise = tf.random.normal([batch_size, latent_dim])
    random_label_indices = tf.random.uniform([batch_size], 0, num_classes, dtype=tf.int32)
    random_labels_one_hot = tf.one_hot(random_label_indices, depth=num_classes)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 가짜 이미지 생성
        fake_images = generator([noise, random_labels_one_hot], training=True)

        # 디스크리미네이터 예측
        real_validity, real_class = discriminator(real_images, training=True)
        fake_validity, fake_class = discriminator(fake_images, training=True)

        # 출력 형태 및 데이터 타입 확인 및 조정
        real_validity = tf.cast(tf.reshape(real_validity, [-1]), tf.float32)
        fake_validity = tf.cast(tf.reshape(fake_validity, [-1]), tf.float32)

        # 라벨의 데이터 타입 변환
        real_labels = tf.cast(real_labels, tf.int32)

        # 손실 계산을 위한 라벨 생성
        ones = tf.ones_like(real_validity)
        zeros = tf.zeros_like(fake_validity)

        # 제너레이터 손실
        adversarial_loss = bce_loss(ones, fake_validity)
        mutual_info_loss = scce_loss(random_label_indices, fake_class)
        gen_loss = adversarial_loss + lambda_coeff * mutual_info_loss

        # 디스크리미네이터 손실
        disc_loss_real = bce_loss(ones, real_validity)
        disc_loss_fake = bce_loss(zeros, fake_validity)
        disc_class_loss_real = scce_loss(real_labels, real_class)
        disc_class_loss_fake = scce_loss(random_label_indices, fake_class)
        mutual_info_loss_disc = disc_class_loss_real + disc_class_loss_fake

        disc_loss = disc_loss_real + disc_loss_fake + lambda_coeff * mutual_info_loss_disc

    # 그래디언트 계산 및 적용
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_loss, disc_loss


# 학습 루프
epochs = 10
for epoch in range(epochs):
    print(f'Start of epoch {epoch+1}')
    for step, (real_images, real_labels) in enumerate(tf_dataset):

        gen_loss, disc_loss = train_step(real_images, real_labels)

        if step % 100 == 0:
            print(f'Epoch {epoch+1}, Step {step}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}')