import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tensorflow.keras.preprocessing import image_dataset_from_directory


# 하이퍼파라미터
batch_size = 16
image_height = 256
image_width = 128
channels = 3
latent_dim = 100
categorical_dim = 10  # 예: 10개의 범주
continuous_dim = 2    # 예: 2개의 연속 변수
total_dim = latent_dim + categorical_dim + continuous_dim
buffer_size = 1000
num_epochs = 50
sample_interval = 100

# 데이터 디렉토리 설정 (이미지가 저장된 디렉토리 경로로 변경)
data_dir = './data/images'  # 예: './data/images' 디렉토리에 이미지가 저장되어 있다고 가정

# 이미지 데이터셋 로드 및 전처리
train_dataset = image_dataset_from_directory(
    data_dir,
    labels=None,  # 레이블이 없는 경우
    label_mode=None,
    batch_size=batch_size,
    image_size=(image_height, image_width),
    shuffle=True,
    seed=123
)

# 데이터 정규화: [-1, 1] 범위로 정규화
normalization_layer = layers.Rescaling(1./127.5, offset=-1)
train_dataset = train_dataset.map(lambda x: normalization_layer(x))
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def build_generator(input_dim, output_shape=(256, 128, 3)):
    model = models.Sequential(name="Generator")
    
    model.add(layers.Dense(8*4*256, use_bias=False, input_dim=input_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Reshape((8, 4, 256)))  # 시작 크기: (8, 4, 256)
    assert model.output_shape == (None, 8, 4, 256)  # 배치 크기는 None
    
    # Upsampling to (16, 8, 128)
    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    assert model.output_shape == (None, 16, 8, 128)
    
    # Upsampling to (32, 16, 64)
    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    assert model.output_shape == (None, 32, 16, 64)
    
    # Upsampling to (64, 32, 32)
    model.add(layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    assert model.output_shape == (None, 64, 32, 32)
    
    # Upsampling to (128, 64, 16)
    model.add(layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    assert model.output_shape == (None, 128, 64, 16)
    
    # Upsampling to (256, 128, 3)
    model.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False))
    model.add(layers.Activation('tanh'))  # 출력 범위: [-1, 1]
    assert model.output_shape == (None, 256, 128, 3)
    
    return model


def build_discriminator_with_q(input_shape=(256, 128, 3), categorical_dim=10, continuous_dim=2):
    input_img = layers.Input(shape=input_shape)
    
    # Downsampling to (128, 64, 64)
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(input_img)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Downsampling to (64, 32, 128)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Downsampling to (32, 16, 256)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Downsampling to (16, 8, 512)
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Flatten
    x = layers.Flatten()(x)
    features = layers.Dense(1024)(x)
    features = layers.BatchNormalization()(features)
    features = layers.LeakyReLU(alpha=0.1)(features)
    
    # Discriminator 출력 (진짜/가짜)
    D_output = layers.Dense(2, activation='softmax', name='D_output')(features)
    
    # Q-Network 출력
    # 범주형 변수에 대한 로짓
    q_logits = layers.Dense(categorical_dim, name='c_logits')(features)
    
    # 연속 변수에 대한 평균
    q_mu = layers.Dense(continuous_dim, name='s_mu')(features)
    
    # 연속 변수에 대한 분산 (양수 유지)
    q_var = layers.Dense(continuous_dim, activation='softplus', name='s_var')(features)
    
    model = models.Model(inputs=input_img, outputs=[D_output, q_logits, q_mu, q_var], name='Discriminator_with_Q')
    return model


# 손실 함수
cross_entropy = losses.CategoricalCrossentropy(from_logits=False)  # 소프트맥스 출력 사용
categorical_cross_entropy = losses.CategoricalCrossentropy(from_logits=False)
mse_loss = losses.MeanSquaredError()

# 모델 초기화
generator = build_generator(latent_dim + categorical_dim + continuous_dim)
discriminator = build_discriminator_with_q()

# 최적화기
lr = 2e-4
beta1 = 0.5
beta2 = 0.999

optimizer_G = optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
optimizer_D = optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)


# 잠재 변수 샘플링 함수
def sample_latent(batch_size):
    # 잠재 변수 z (정규 분포)
    z = np.random.randn(batch_size, latent_dim)
    
    # 범주형 변수 c (One-hot)
    c = np.random.randint(0, categorical_dim, batch_size)
    c_onehot = np.zeros((batch_size, categorical_dim))
    c_onehot[np.arange(batch_size), c] = 1
    
    # 연속 변수 s (균일 분포 [-1, 1])
    s = np.random.uniform(-1, 1, (batch_size, continuous_dim))
    
    latent = np.concatenate([z, c_onehot, s], axis=1)
    return latent, c, s

# 훈련 단계 정의
@tf.function
def train_step(real_images):
    batch_size_current = tf.shape(real_images)[0]
    
    # 잠재 변수 샘플링
    latent, c, s = sample_latent(batch_size_current)
    latent = tf.convert_to_tensor(latent, dtype=tf.float32)
    c_onehot = tf.one_hot(c, depth=categorical_dim)
    s = tf.convert_to_tensor(s, dtype=tf.float32)
    
    # 진짜와 가짜 레이블 (원-핫 인코딩)
    real_labels = tf.one_hot([0] * batch_size_current, depth=2)  # 진짜: [1, 0]
    fake_labels = tf.one_hot([1] * batch_size_current, depth=2)  # 가짜: [0, 1]
    
    with tf.GradientTape(persistent=True) as tape:
        # 진짜 이미지에 대한 Discriminator 출력
        D_real, _, _, _ = discriminator(real_images, training=True)
        loss_real = cross_entropy(real_labels, D_real)
        
        # 가짜 이미지 생성
        fake_images = generator(latent, training=True)
        
        # 가짜 이미지에 대한 Discriminator 출력
        D_fake, q_logits_fake, q_mu_fake, q_var_fake = discriminator(fake_images, training=True)
        loss_fake = cross_entropy(fake_labels, D_fake)
        
        # 총 Discriminator 손실
        loss_D = loss_real + loss_fake
        
        # Generator 손실 (Discriminator가 가짜 이미지를 진짜로 분류하도록 유도)
        loss_G = cross_entropy(real_labels, D_fake)
        
        # Q-Network 손실
        # 범주형 변수에 대한 손실
        loss_c = categorical_cross_entropy(c_onehot, q_logits_fake)
        
        # 연속 변수에 대한 손실 (MSE)
        loss_s = mse_loss(s, q_mu_fake)
        
        loss_Q = loss_c + loss_s
        
        # 총 Generator 손실
        loss_G_total = loss_G + loss_Q  # 하이퍼파라미터 조정 가능
    
    # Discriminator의 그래디언트 계산 및 적용
    gradients_D = tape.gradient(loss_D, discriminator.trainable_variables)
    optimizer_D.apply_gradients(zip(gradients_D, discriminator.trainable_variables))
    
    # Generator의 그래디언트 계산 및 적용
    gradients_G = tape.gradient(loss_G_total, generator.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, generator.trainable_variables))
    
    return loss_D, loss_G, loss_Q

# 학습 루프
for epoch in range(num_epochs):
    for step, real_images in enumerate(train_dataset):
        loss_D, loss_G, loss_Q = train_step(real_images)
        
        # 진행 상황 출력
        if step % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step {step} \
                  Loss D: {loss_D.numpy():.4f}, Loss G: {loss_G.numpy():.4f}, Loss Q: {loss_Q.numpy():.4f}")
    
    # 에포크 끝날 때마다 샘플 이미지 저장
    if (epoch + 1) % 1 == 0:
        num_samples = 10
        # 잠재 변수 생성 (c는 서로 다른 클래스, s는 0)
        z = np.random.randn(num_samples, latent_dim)
        c_fixed = np.eye(categorical_dim)[:num_samples]
        s_fixed = np.zeros((num_samples, continuous_dim))
        latent_fixed = np.concatenate([z, c_fixed, s_fixed], axis=1)
        latent_fixed = tf.convert_to_tensor(latent_fixed, dtype=tf.float32)
        
        generated_images = generator(latent_fixed, training=False)
        generated_images = (generated_images + 1.0) / 2.0  # [0,1] 범위로 변환
        
        # 이미지 시각화
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
        for img, ax in zip(generated_images, axes):
            ax.imshow(img.numpy())
            ax.axis('off')
        plt.show()
