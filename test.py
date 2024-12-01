import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose, BatchNormalization, LeakyReLU, Conv2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from preprocess import parse_and_process_file
from preprocess import create_optimized_dataset
import matplotlib.pyplot as plt
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

# 데이터 로딩 및 전처리 함수 정의
def load_and_preprocess_image(file_path, label):
    # .mat 파일 로드 (예시로 이미지 로딩으로 대체)
    # 실제 데이터 로딩 코드를 여기에 작성하세요
    # 예시로 랜덤 이미지 생성
    image = tf.random.uniform([256, 128, 3], 0, 1)
    return image, label

batch_size = 16
nperseg = 1024
noverlap = 768
original_shape = (512,256)
downsampled_shape = (256, 128)  # Downsampled shape
buffer_size = 100
num_classes = 2
# TensorFlow 데이터셋 생성
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
dataset = dataset = dataset.map(
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
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# 제너레이터 모델 정의
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

# 디스크리미네이터 모델 정의
def build_discriminator(input_shape=(256, 128, 3), num_classes=2):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    validity = Dense(1, activation='sigmoid')(x)
    label = Dense(num_classes, activation='softmax')(x)

    discriminator = Model(inputs, [validity, label])
    return discriminator

# 모델 생성
latent_dim = 100
num_classes = 2

generator = build_generator(latent_dim, num_classes)
discriminator = build_discriminator()

# 옵티마이저와 손실 함수 정의
gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
disc_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

bce_loss = tf.keras.losses.BinaryCrossentropy()
scce_loss = tf.keras.losses.CategoricalCrossentropy()

# 학습 단계 함수 정의
# @tf.function
def train_step(real_images, real_labels):
    batch_size = tf.shape(real_images)[0]

    noise = tf.random.normal([batch_size, latent_dim])
    random_labels = tf.random.uniform([batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
    random_labels_one_hot = tf.one_hot(random_labels, num_classes)

    with tf.GradientTape(persistent=True) as tape:
        # 가짜 이미지 생성
        fake_images = generator([noise, random_labels_one_hot], training=True)

        # 진짜와 가짜 이미지 구분
        real_validity, real_pred_label = discriminator(real_images, training=True)
        fake_validity, fake_pred_label = discriminator(fake_images, training=True)

        # 손실 계산
        real_labels_float = tf.ones_like(real_validity)
        fake_labels_float = tf.zeros_like(fake_validity)

        d_loss_real = bce_loss(real_labels_float, real_validity)
        d_loss_fake = bce_loss(fake_labels_float, fake_validity)
        d_loss_class_real = scce_loss(real_labels, real_pred_label)
        d_loss_class_fake = scce_loss(random_labels_one_hot, fake_pred_label)
        d_loss = d_loss_real + d_loss_fake + d_loss_class_real + d_loss_class_fake

        g_loss_adv = bce_loss(real_labels_float, fake_validity)
        g_loss_class = scce_loss(random_labels_one_hot, fake_pred_label)
        g_loss = g_loss_adv + g_loss_class

    # 그래디언트 계산 및 적용
    grads_gen = tape.gradient(g_loss, generator.trainable_variables)
    grads_disc = tape.gradient(d_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_variables))

    return g_loss, d_loss

# 학습 루프
epochs = 30
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for step, (real_images, real_labels) in enumerate(dataset):
        g_loss, d_loss = train_step(real_images, real_labels)

        # if step % 100 == 0:
        print(f"Step {step}, Generator Loss: {g_loss.numpy()}, Discriminator Loss: {d_loss.numpy()}")

print("학습 완료!")

# infoGAN 모델이 이미 학습되어 있다고 가정합니다.

def generate_infoGAN_data(generator, num_samples, latent_dim, num_classes, downsampled_shape):
    # 노이즈와 랜덤 레이블 생성
    noise = tf.random.normal([num_samples, latent_dim])
    random_label_indices = tf.random.uniform([num_samples], 0, num_classes, dtype=tf.int32)
    random_labels = tf.one_hot(random_label_indices, depth=num_classes)

    # 이미지 생성
    generated_images = generator([noise, random_labels], training=False)

    # 이미지 크기 조정 (downsampled_shape로 변경)
    generated_images_resized = tf.image.resize(generated_images, downsampled_shape)

    # 이미지 정규화 (필요한 경우)
    # generated_images_resized = generated_images_resized / 255.0

    # 레이블은 원-핫 인코딩된 상태
    return generated_images_resized, random_labels

# 필요한 수의 샘플 생성 (예: 5000개)
num_generated_samples = 10
latent_dim = 100  # infoGAN 생성자에서 사용한 latent_dim
num_classes = 2   # 분류할 클래스 수

generated_images, generated_labels = generate_infoGAN_data(generator, num_generated_samples, latent_dim, num_classes, downsampled_shape)


def plot_generated_images(images, labels, num_classes=2, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        
        plt.imshow(images[i])
        plt.title(f"Class: {np.argmax(labels[i])}")
        plt.axis('off')
    plt.show()

plot_generated_images(generated_images, generated_labels, num_classes, num_samples=5)
