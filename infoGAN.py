import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tensorflow.keras.preprocessing import image_dataset_from_directory

from makeDataset import createTfDataset

# 하이퍼파라미터
batch_size = 16
image_height = 256
image_width = 128
channels = 3
latent_dim = 100
categorical_dim = 2  # 예: 2개의 범주
continuous_dim = 0    # 예: 0개의 연속 변수
total_dim = latent_dim + categorical_dim + continuous_dim
buffer_size = 1000
num_epochs = 50
sample_interval = 100
lambda_info = 1.0  # InfoGAN의 상호 정보 손실 가중치


# 데이터 디렉토리 설정 (이미지가 저장된 디렉토리 경로로 변경)
data_dir = './data/images'  # 예: './data/images' 디렉토리에 이미지가 저장되어 있다고 가정

visualize = False
save = True
# 이미지 데이터셋 로드 및 전처리
train_dataset = createTfDataset()

# 데이터 정규화: [-1, 1] 범위로 정규화
# normalization_layer = layers.Rescaling(1./127.5, offset=-1)
# train_dataset = train_dataset.map(lambda x: normalization_layer(x))
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# 생성기 모델 정의
def build_generator(input_dim, output_shape=(256, 128, 3)):
    model = models.Sequential(name="Generator")
    
    model.add(layers.Dense(8*4*256, use_bias=False, input_dim=input_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Reshape((8, 4, 256)))  # 시작 크기: (8, 4, 256)
    
    # Upsampling 단계들
    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # 최종 출력 레이어
    model.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.Activation('sigmoid'))  # 출력 범위: [0, 1]
    
    return model

# 판별기 모델 정의
def build_discriminator_with_q(input_shape=(256, 128, 3), categorical_dim=2, continuous_dim=0):
    input_img = layers.Input(shape=input_shape)
    
    # Downsampling 단계들
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(input_img)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Flatten 및 특징 추출
    x = layers.Flatten()(x)
    features = layers.Dense(1024)(x)
    features = layers.BatchNormalization()(features)
    features = layers.LeakyReLU(alpha=0.1)(features)
    
    # Discriminator 출력 (진짜/가짜)
    D_output = layers.Dense(2, activation='softmax', name='D_output')(features)
    
    # Q-Network 출력 (범주형 변수)
    q_logits = layers.Dense(categorical_dim, name='c_logits')(features)
    
    # 연속 변수 출력 (필요 시)
    if continuous_dim > 0:
        q_mu = layers.Dense(continuous_dim, name='s_mu')(features)
        q_var = layers.Dense(continuous_dim, activation='softplus', name='s_var')(features)
        outputs = [D_output, q_logits, q_mu, q_var]
    else:
        outputs = [D_output, q_logits]
    
    return models.Model(inputs=input_img, outputs=outputs, name='Discriminator_with_Q')

# 손실 함수 정의
cross_entropy = losses.CategoricalCrossentropy(from_logits=False)  # 소프트맥스 출력 사용
categorical_cross_entropy = losses.CategoricalCrossentropy(from_logits=False)
mse_loss = losses.MeanSquaredError()

# 모델 초기화
generator = build_generator(latent_dim + categorical_dim + continuous_dim)
discriminator = build_discriminator_with_q(input_shape=(256, 128, 3), categorical_dim=categorical_dim, continuous_dim=continuous_dim)

# 최적화기 정의
lr = 2e-4
beta1 = 0.5
beta2 = 0.999

optimizer_G = optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
optimizer_D = optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)

# 잠재 변수 샘플링 함수
def sample_latent(batch_size):
    # 잠재 변수 z (정규 분포)
    z = tf.random.normal([batch_size, latent_dim])
    
    # 범주형 변수 c (One-hot)
    c = tf.random.uniform([batch_size], minval=0, maxval=categorical_dim, dtype=tf.int32)
    c_onehot = tf.one_hot(c, depth=categorical_dim)
    
    # 연속 변수 s (연속 변수 없음)
    if continuous_dim > 0:
        s = tf.random.uniform([batch_size, continuous_dim], minval=-1, maxval=1)
        latent = tf.concat([z, c_onehot, s], axis=1)
    else:
        latent = tf.concat([z, c_onehot], axis=1)
    
    return latent, c

# 훈련 단계 정의
# @tf.function
def train_step(real_images):
    # 고정된 배치 크기 사용
    batch_size_current = batch_size  # 정수로 고정
    
    # 잠재 변수 샘플링
    latent, c = sample_latent(batch_size_current)
    latent = tf.convert_to_tensor(latent, dtype=tf.float32)
    c_onehot = tf.one_hot(c, depth=categorical_dim)
    
    # 진짜와 가짜 레이블 (원-핫 인코딩)
    real_labels = tf.one_hot([0] * batch_size_current, depth=2)  # 진짜: [1, 0]
    fake_labels = tf.one_hot([1] * batch_size_current, depth=2)  # 가짜: [0, 1]
    
    # **1. 판별기 손실 (loss_D + lambda * loss_Q)**
    with tf.GradientTape() as tape_D:
        # 진짜 이미지에 대한 Discriminator 출력
        D_real_outputs = discriminator(real_images, training=True)
        D_real = D_real_outputs[0]
        loss_real = cross_entropy(real_labels, D_real)
        
        # 가짜 이미지 생성
        fake_images = generator(latent, training=True)
        
        # 가짜 이미지에 대한 Discriminator 출력
        D_fake_outputs = discriminator(fake_images, training=True)
        D_fake = D_fake_outputs[0]
        loss_fake = cross_entropy(fake_labels, D_fake)
        
        # 총 Discriminator 손실
        loss_D = loss_real + loss_fake
        
        # Q-Network 손실 (mutual information)
        q_logits_fake = D_fake_outputs[1]
        loss_c = categorical_cross_entropy(c_onehot, q_logits_fake)
        
        if continuous_dim > 0:
            q_mu_fake = D_fake_outputs[2]
            q_var_fake = D_fake_outputs[3]
            # 실제 연속 변수 s를 샘플링
            s = tf.random.uniform([batch_size_current, continuous_dim], minval=-1, maxval=1)
            loss_s = mse_loss(s, q_mu_fake)
            loss_Q = loss_c + loss_s
        else:
            loss_Q = loss_c  # 연속 변수가 없을 경우
        
        # 총 Discriminator 손실에 InfoGAN의 Q-Network 손실 추가
        loss_D_total = loss_D + lambda_info * loss_Q
    
    # 판별기의 그래디언트 계산 및 적용
    gradients_D = tape_D.gradient(loss_D_total, discriminator.trainable_variables)
    optimizer_D.apply_gradients(zip(gradients_D, discriminator.trainable_variables))
    
    # **2. 생성기 손실 (loss_G)**
    with tf.GradientTape() as tape_G:
        # 가짜 이미지 생성
        fake_images = generator(latent, training=True)
        
        # 가짜 이미지에 대한 Discriminator 출력
        D_fake_outputs = discriminator(fake_images, training=True)
        D_fake = D_fake_outputs[0]
        
        # 생성기의 손실: Discriminator가 가짜 이미지를 진짜로 분류하도록 유도
        loss_G = cross_entropy(real_labels, D_fake)
    
    # 생성기의 그래디언트 계산 및 적용
    gradients_G = tape_G.gradient(loss_G, generator.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, generator.trainable_variables))
    
    return loss_D, loss_G, loss_Q

# 학습 루프
for epoch in range(num_epochs):
    for step, (real_images, _) in enumerate(train_dataset):
        loss_D, loss_G, loss_Q = train_step(real_images)
        
        # 진행 상황 출력
        # if step % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Step {step} \
                Loss D: {loss_D.numpy():.4f}, Loss G: {loss_G.numpy():.4f}, Loss Q: {loss_Q.numpy():.4f}")
    
    # 에포크 끝날 때마다 샘플 이미지 저장
    if (epoch + 1) % 1 == 0:
        num_samples = 10
        # 잠재 변수 생성 (c는 서로 다른 클래스, s는 0)
        z = np.random.randn(num_samples, latent_dim)
        c_fixed = np.tile(np.eye(categorical_dim), (int(np.ceil(num_samples / categorical_dim)), 1))[:num_samples]
        if continuous_dim >0:
            s_fixed = np.zeros((num_samples, continuous_dim))
            latent_fixed = np.concatenate([z, c_fixed, s_fixed], axis=1)
        else:
            latent_fixed = np.concatenate([z, c_fixed], axis=1)
        latent_fixed = tf.convert_to_tensor(latent_fixed, dtype=tf.float32)
        
        generated_images = generator(latent_fixed, training=False)
        generated_images = generated_images.numpy()  # [0, 1] 범위로 유지

        # 시각화용: [0, 1] 범위를 그대로 사용
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
        for img, ax in zip(generated_images, axes):
            ax.imshow(img)  # [0, 1] 범위 사용
            ax.axis('off')

        # 이미지 시각화
        if visualize:
            plt.show()

        # 저장용: [0, 255] 범위로 변환
        if save:
            save_images = (generated_images * 255).astype(np.uint8)  # [0, 255]로 변환
            # save_path = os.path.join("generatedImage", f"generated_image_epoch{epoch+1}.png")
            
                
            for img_idx, img in enumerate(save_images):  # 각 이미지에 대해
                # 채널 분리 (R, G, B)
                for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
                    channel = img[:, :, channel_idx]  # (256, 128)
                    save_path = os.path.join("generatedImage", f"generated_image_{img_idx}_{channel_name}.png")
                    plt.imsave(save_path, channel, cmap='gray')  # 흑백으로 저장
                    print(f"Saved {channel_name} channel at: {save_path}")
            plt.close(fig)  # 현재 figure 닫기
