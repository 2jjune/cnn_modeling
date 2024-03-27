import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import math

se_ratio = 4
expand_ratio = 6
width_coefficient = 1.0
depth_coefficient = 1.0
default_resolution = 112
input_channels = 3
depth_divisor= 8 
dropout_rate = 0.2
drop_connect_rate = 0.2

kernel_size = [3,3,5,3,5,5,3]
num_repeat = [1,2,2,3,3,4,1]
output_filters = [16,24,40,80,112,192,320]
# strides = [1,2,1,2,1,2,1]
strides = [1,2,2,2,1,2,1]
MBConvBlock_1_True  =  [True,False,False,False,False,False,False]

def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))

def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

class DropConnect(layers.Layer):
    def __init__(self, drop_connect_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training):
        def _drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += K.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = tf.math.divide(inputs, keep_prob) * binary_tensor
            return output

        return K.in_train_phase(_drop_connect, inputs, training=training)
    
def SEBlock(filters,reduced_filters):
    def _block(inputs):
        x = layers.GlobalAveragePooling2D()(inputs)
        x = layers.Reshape((1,1,x.shape[1]))(x)
        x = layers.Conv2D(reduced_filters, 1, 1)(x)
        x = tfa.activations.mish(x)
        x = layers.Conv2D(filters, 1, 1)(x)
        x = layers.Activation('sigmoid')(x)
        x = layers.Multiply()([x, inputs])
        return x
    return _block

def MBConvBlock(x,kernel_size, strides,drop_connect_rate,output_channels,MBConvBlock_1_True=False):
    output_channels = round_filters(output_channels,width_coefficient,depth_divisor)
    if MBConvBlock_1_True:
        block = layers.DepthwiseConv2D(kernel_size, strides,padding='same', use_bias=False)(x)
        block = layers.BatchNormalization()(block)
        block = tfa.activations.mish(block)
        block = SEBlock(x.shape[3],x.shape[3]/se_ratio)(block)
        block = layers.Conv2D(output_channels, (1,1), padding='same', use_bias=False)(block)
        block = layers.BatchNormalization()(block)
        return block

    channels = x.shape[3]
    expand_channels = channels * expand_ratio
    block = layers.Conv2D(expand_channels, (1,1), padding='same', use_bias=False)(x)
    block = layers.BatchNormalization()(block)
    block = tfa.activations.mish(block)
    block = layers.DepthwiseConv2D(kernel_size, strides,padding='same', use_bias=False)(block)
    block = layers.BatchNormalization()(block)
    block = tfa.activations.mish(block)
    block = SEBlock(expand_channels,channels/se_ratio)(block)
    block = layers.Conv2D(output_channels, (1,1), padding='same', use_bias=False)(block)
    block = layers.BatchNormalization()(block)
    if x.shape[3] == output_channels:
        block = DropConnect(drop_connect_rate)(block)
        block = layers.Add()([block, x])
    return block

def Build_Network(num_classes, booster=False):
    x_input = layers.Input(shape=(default_resolution,default_resolution,input_channels))    
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), (3,3), 2,padding='same', use_bias=False)(x_input)
    x = layers.BatchNormalization()(x)
    x = tfa.activations.mish(x)
    num_blocks_total = sum(num_repeat)
    block_num = 0
    for i in range(len(kernel_size)):
        round_num_repeat = round_repeats(num_repeat[i], depth_coefficient)
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = MBConvBlock(x,kernel_size[i],strides[i],drop_rate,output_filters[i],MBConvBlock_1_True = MBConvBlock_1_True[i])
        block_num += 1
        if round_num_repeat > 1:
            for bidx in range(round_num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                x = MBConvBlock(x,kernel_size[i],1,drop_rate,output_filters[i],MBConvBlock_1_True = MBConvBlock_1_True[i])
                block_num += 1
    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,padding='same',use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tfa.activations.mish(x)
    if booster==False:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(num_classes,activation='softmax')(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x)
    return model