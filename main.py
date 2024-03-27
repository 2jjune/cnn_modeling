import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, ReLU, Add, MaxPooling2D, Dropout, Multiply, Reshape
from tensorflow.keras.models import Model
import pathlib

base_dir = 'D:/dataset/cifar10/'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

IMG_SIZE = (224, 224)
# generator = ImageDataGenerator(rotation_range=40,
#                                width_shift_range=0.2,
#                                height_shift_range=0.2,
#                                #                           shear_range=0.2,
#                                #                           zoom_range=0.2,
#                                horizontal_flip=True,
#                                #                           vertical_flip=True,
#                                fill_mode='reflect',
#                                validation_split=0.3,
#                                rescale=1 / 255.)
#
# test_generator = ImageDataGenerator(rescale=1 / 255.)
# # print(help(ImageDataGenerator))
#
# train_gen = generator.flow_from_directory(
#     train_dir,
#     target_size=IMG_SIZE,
#     class_mode='categorical',
#     shuffle=True,
#     subset='training', )
# val_gen = generator.flow_from_directory(
#     train_dir,
#     target_size=IMG_SIZE,
#     class_mode='categorical',
#     shuffle=False,
#     subset='validation', )
# test_gen = test_generator.flow_from_directory(
#     test_dir,
#     target_size=IMG_SIZE,
#     #             class_mode='categorical',
#     shuffle=False, )
# 클래스 이름을 저장할 빈 리스트를 생성합니다.
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(224, 224),
  batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(224, 224),
  batch_size=32)
test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=(224, 224),
  batch_size=32)
# import tensorflow_datasets as tfds
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1. / 255),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomBrightness(factor=0.1),
    
])

print(train_ds)

# AUTOTUNE = tf.data.AUTOTUNE
#
# def prepare(ds, shuffle=False, augment=False):
#   # Resize and rescale all datasets.
#   ds = ds.map(lambda x, y: (data_augmentation(x), y),
#               num_parallel_calls=AUTOTUNE)
#
#   if shuffle:
#     ds = ds.shuffle(1000)
#
#   # Batch all datasets.
#   # ds = ds.batch(32)
#
#   # Use data augmentation only on the training set.
#   if augment:
#     ds = ds.map(lambda x, y: (data_augmentation(x), y),
#                 num_parallel_calls=AUTOTUNE)
#
#   # Use buffered prefetching on all datasets.
#   return ds.prefetch(buffer_size=AUTOTUNE)
#
# train_ds = prepare(train_ds, shuffle=True, augment=True)
# val_ds = prepare(val_ds)
# test_ds = prepare(test_ds)

# base_model = tf.keras.applications.EfficientNetB0(include_top=False,
#                                                   weights='imagenet',
#                                                   input_shape=(224, 224, 3))
#
# # 기본 모델의 층을 고정 (가중치를 업데이트하지 않음)
# base_model.trainable = False
#
# # 모델 구축
# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),  # 특징 맵을 하나의 특징 벡터로 변환
#     layers.Dense(512, activation='relu'),  # 새로운 분류 층
#     #     layers.Dropout(0.2),
#     layers.Dense(7, activation='softmax')  # 가정: 10개의 클래스가 있다
# ])

def SEBlock(input_feature, ratio=16):
    """Squeeze and Excitation Block."""
    channel_axis = -1
    channel = input_feature.shape[channel_axis]

    se_shape = (1, 1, channel)
    se = GlobalAveragePooling2D()(input_feature)
    se = Dense(channel // ratio, activation='relu', use_bias=False)(se)
    se = Dense(channel, activation='sigmoid', use_bias=False)(se)
    se = Reshape(se_shape)(se)
    return Multiply()([input_feature, se])


def conv_and_se_block(x, filters):
    """A helper function to apply Convolution + SE Block."""
    x = Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SEBlock(x)
    return x


def depthwise_sep_conv_and_se_block(x, filters):
    """A helper function to apply Depthwise Separable Conv + SE Block."""
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SEBlock(x)
    return x


def make_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    inputs = data_augmentation(inputs)
    x = conv_and_se_block(inputs, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = depthwise_sep_conv_and_se_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = depthwise_sep_conv_and_se_block(x, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Additional depth for larger input size
    x = depthwise_sep_conv_and_se_block(x, 256)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = depthwise_sep_conv_and_se_block(x, 512)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# 모델 생성 (예: CIFAR-10의 입력 크기와 클래스 수 사용)
model = make_model((224, 224, 3), 10)
# model.summary()
# base_model = tf.keras.applications.VGG16(include_top=False,
#                                                   weights='imagenet',
#                                                   input_shape=(224, 224, 3))
#
# # 기본 모델의 층을 고정 (가중치를 업데이트하지 않음)
# base_model.trainable = False
#
# # 모델 구축
# model = tf.keras.models.Sequential([
#     data_augmentation,
#     base_model,
#     layers.GlobalAveragePooling2D(),  # 특징 맵을 하나의 특징 벡터로 변환
#     layers.Dense(4096, activation='relu'),  # 새로운 분류 층 tfa.optimizers.mish
#     layers.Dense(4096, activation='relu'),  # 새로운 분류 층
#     #     layers.Dropout(0.2),
#
#     layers.Dense(10, activation='softmax')  # 가정: 10개의 클래스가 있다
# ])

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for multi-class or categorical classification.
    This loss function generalizes the idea of Focal Loss to multi-class classification.
    """

    def focal_loss_fixed(y_true, y_pred):
        """
        Focal loss calculation.
        """
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return tf.reduce_sum(loss, axis=1)

    return focal_loss_fixed


callback = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=7, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4)]
def Ranger(sync_period=6,
           slow_step_size=0.5,
           learning_rate=0.001,
           beta_1=0.9,
           beta_2=0.999,
           epsilon=1e-7,
           weight_decay=0.,
           amsgrad=False,
           sma_threshold=5.0,
           total_steps=0,
           warmup_proportion=0.1,
           min_lr=0.,
           name="Ranger"):
    inner = tfa.optimizers.RectifiedAdam(learning_rate, beta_1, beta_2, epsilon, weight_decay, amsgrad, sma_threshold, total_steps, warmup_proportion, min_lr, name)
    optim = tfa.optimizers.Lookahead(inner, sync_period, slow_step_size, name)
    return optim


model.compile(
    # optimizer='adam',
    optimizer=Ranger(),
    #     optimizer=tf.keras.optimizers.Adam(0.001),
    #     optimizer=tfa.optimizers.AdamW(0.001),
    # loss=categorical_focal_loss(gamma=2., alpha=0.25),
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=80,
          callbacks=callback
          )

# model.save('my_model.h5')
# predictions = model.predict(test_ds)

# 예측 결과를 CSV 파일로 저장
# 이미지 인덱스와 최대 예측값(가장 높은 확률을 가진 클래스)을 매핑
# predicted_classes = tf.argmax(predictions, axis=1).numpy()
# image_indices = range(len(predicted_classes))
# prediction_results = pd.DataFrame({'Image_Index': image_indices, 'Predicted_Class': predicted_classes})
#
# # CSV 파일로 저장
# prediction_results.to_csv('prediction_results.csv', index=False)


from tensorflow.keras.preprocessing import image
####이미지 하나씩읽어와서 파일이름과 예측값 저장
# image_size = (224, 224)
#
# def load_and_preprocess_image(path):
#     img = image.load_img(path, target_size=image_size)
#     img_array = image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Model expects a batch dimension
#     return img_array / 255.
#
# model = tf.keras.models.load_model('my_model.h5')
#
# # Modify this to filter out directories
# image_paths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, fname))]
#
# predictions = []
# for path in image_paths:
#     img_array = load_and_preprocess_image(path)
#     pred = model.predict(img_array)
#     predicted_class = np.argmax(pred, axis=1)[0]
#     predictions.append(predicted_class)
#
# # Save the image file names and their predicted classes
# results_df = pd.DataFrame({
#     'Image_Name': [os.path.basename(path) for path in image_paths],
#     'Predicted_Class': predictions
# })
#
# results_df.to_csv('image_predictions.csv', index=False)

model = tf.keras.models.load_model('my_model.h5')

# 전체 데이터셋에 대해 예측 수행
predictions = model.predict(test_ds)

# 가장 높은 확률을 가진 클래스 예측
predicted_classes = np.argmax(predictions, axis=1)

# 이미지 파일 이름 추출
filenames = test_ds.file_paths  # image_dataset_from_directory를 사용할 경우 file_paths 속성 참조

# 이미지 파일 이름과 예측값을 DataFrame으로 저장
results_df = pd.DataFrame({
    'Image_Name': [os.path.basename(filename) for filename in filenames],
    'Predicted_Class': predicted_classes
})

# 결과를 CSV 파일로 저장
results_df.to_csv('batch_image_predictions.csv', index=False)