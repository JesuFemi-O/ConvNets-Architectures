import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from time import time
import numpy as np

from keras.utils.vis_utils import model_to_dot

def visualize_model(model):
    """
    Simple Function visualize model architectures
    
    Args:
    
    model -> the instantiated model object
    """
    K.clear_session()
    
    return SVG(model_to_dot(model).create(prog='dot', format='svg'))



def alexnet(input_shape, n_classes):
    input = Input(input_shape)

    # actually batch normalization didn't exist back then
    # they used LRN (Local Response Normalization) for regularization
    x = Conv2D(96, 11, strides=4, padding='same', activation='relu')(input)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2)(x)

    x = Conv2D(256, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2)(x)

    x = Conv2D(384, 3, strides=1, padding='same', activation='relu')(x)

    x = Conv2D(384, 3, strides=1, padding='same', activation='relu')(x)

    x = Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2)(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)

    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model

def VGG(input_shape, n_classes):
    input = Input(input_shape)
    
    x = Conv2D(64, 3, padding='same', activation='relu')(input)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPool2D(2, strides=2, padding='same')(x)
    
    x = Conv2D(128, 3, padding='same',activation='relu')(x)
    x = Conv2D(128, 3, padding='same',activation='relu')(x)
    x = MaxPool2D(2, strides=2, padding='same')(x)
    
    x = Conv2D(256, 3, padding='same',activation='relu')(x)
    x = Conv2D(256, 3, padding='same',activation='relu')(x)
    x = Conv2D(256, 3, padding='same',activation='relu')(x)
    x = MaxPool2D(2, strides=2, padding='same')(x)
    
    x = Conv2D(512, 3, padding='same',activation='relu')(x)
    x = Conv2D(512, 3, padding='same',activation='relu')(x)
    x = Conv2D(512, 3, padding='same',activation='relu')(x)
    x = MaxPool2D(2, strides=2, padding='same')(x)
    
    x = Conv2D(512, 3, padding='same',activation='relu')(x)
    x = Conv2D(512, 3, padding='same',activation='relu')(x)
    x = Conv2D(512, 3, padding='same',activation='relu')(x)
    x = MaxPool2D(2, strides=2, padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    output = Dense(n_classes, activation='softmax')(x)
    
    model = Model(input, output)
    return model


def googlenet(input_shape, n_classes):
  
    def inception_block(x, f):
        t1 = Conv2D(f[0], 1, activation='relu')(x)

        t2 = Conv2D(f[1], 1, activation='relu')(x)
        t2 = Conv2D(f[2], 3, padding='same', activation='relu')(t2)

        t3 = Conv2D(f[3], 1, activation='relu')(x)
        t3 = Conv2D(f[4], 5, padding='same', activation='relu')(t3)

        t4 = MaxPool2D(3, 1, padding='same')(x)
        t4 = Conv2D(f[5], 1, activation='relu')(t4)

        output = Concatenate()([t1, t2, t3, t4])
        return output


    input = Input(input_shape)

    x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = Conv2D(64, 1, activation='relu')(x)
    x = Conv2D(192, 3, padding='same', activation='relu')(x)
    x = MaxPool2D(3, strides=2)(x)

    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    x = AvgPool2D(7, strides=1)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model


def mobilenet(input_shape, n_classes):
    
    def mobilenet_block(x, f, s=1):
        x = DepthwiseConv2D(3, strides=s, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(f, 1, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    input = Input(input_shape)

    x = Conv2D(32, 3, strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = mobilenet_block(x, 64)
    x = mobilenet_block(x, 128, 2)
    x = mobilenet_block(x, 128)

    x = mobilenet_block(x, 256, 2)
    x = mobilenet_block(x, 256)

    x = mobilenet_block(x, 512, 2)
    
    for _ in range(5):
        x = mobilenet_block(x, 512)
    
    x = mobilenet_block(x, 1024, 2)
    x = mobilenet_block(x, 1024)

    x = GlobalAvgPool2D()(x)
    
    output = Dense(n_classes, activation='softmax')(x)
    
    model = Model(input, output)
    
    return model

def shufflenet(input_shape, n_classes, g=8):
    channels = 384, 769, 1536
    repetitions = 3, 7, 3
    
    def ch_shuffle(x, g):
#     1 2 3 4 5 6 7 8 9 -reshape-> 1 2 3 -permute dims-> 1 4 7 -reshape-> 1 4 7 2 5 8 3 6 9
#                                  4 5 6                 2 5 8
#                                  7 8 9                 3 6 9
    
        _, w, h, ch = K.int_shape(x)
        ch_g = ch // g
    
        def shuffle_op(x):
            x = K.reshape(x, [-1, w, h, ch_g, g])
            x = K.permute_dimensions(x, [0, 1, 2, 4, 3])
            x = K.reshape(x, [-1, w, h, ch])
            return x

        x = Lambda(shuffle_op)(x)
        return x
    
    def gconv(tensor, ch, g):
        _, _, _, in_ch = K.int_shape(tensor)
        ch_g = in_ch // g
        out_ch = ch // g
        group = []

        for i in range(g):
            x = tensor[:, :, :, i*ch_g:(i+1)*ch_g]
            x = Lambda(lambda x: x[:, :, :, i*ch_g: (i+1)*ch_g])(tensor)
            x = Conv2D(out_ch, 1)(x)
            group.append(x)

        x = Concatenate()(group)
        return x
    
    def shufflenet_block(tensor, ch, s, g):
        x = gconv(tensor, ch // 4, g)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = ch_shuffle(x, g)
        x = DepthwiseConv2D(3, strides=s, padding='same')(x)
        x = BatchNormalization()(x)
        x = gconv(x, ch if s==1 else ch-K.int_shape(tensor)[-1], g)
        x = BatchNormalization()(x)

        if s == 1:
            x = Add()([tensor, x])
        else:
            avg = AvgPool2D(3, strides=2, padding='same')(tensor)
            x = Concatenate()([avg, x])

        output = ReLU()(x)
        return output
    
    def stage(x, ch, r, g):
        x = shufflenet_block(x, ch, 2, g)
        for i in range(r):
            x = shufflenet_block(x, ch, 1, g)
        return x
    
    input = Input(input_shape)
    x = Conv2D(24, 3, strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    for ch, r in zip(channels, repetitions):
        x = stage(x, ch, r, g)

    x = GlobalAvgPool2D()(x)

    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)

    return model

def resnet(input_shape, n_classes):
  
    def conv_bn_rl(x, f, k=1, s=1, p='same'):
        x = Conv2D(f, k, strides=s, padding=p)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
  
  
    def identity_block(tensor, f):
        x = conv_bn_rl(tensor, f)
        x = conv_bn_rl(x, f, 3)
        x = Conv2D(4*f, 1)(x)
        x = BatchNormalization()(x)

        x = Add()([x, tensor])
        output = ReLU()(x)
        return output
  
  
    def conv_block(tensor, f, s):
        x = conv_bn_rl(tensor, f)
        x = conv_bn_rl(x, f, 3, s)
        x = Conv2D(4*f, 1)(x)
        x = BatchNormalization()(x)

        shortcut = Conv2D(4*f, 1, strides=s)(tensor)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        output = ReLU()(x)
        return output
  
  
    def resnet_block(x, f, r, s=2):
        x = conv_block(x, f, s)
        for _ in range(r-1):
            x = identity_block(x, f)
        return x
    
  
    input = Input(input_shape)
  
    x = conv_bn_rl(input, 64, 7, 2)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = resnet_block(x, 64, 3, 1)
    x = resnet_block(x, 128, 4)
    x = resnet_block(x, 256, 6)
    x = resnet_block(x, 512, 3)

    x = GlobalAvgPool2D()(x)

    output = Dense(n_classes, activation='softmax')(x)
  
    model = Model(input, output)
    return model

def densenet(img_shape, n_classes, f=32):
    repetitions = 6, 12, 24, 16
  
    def bn_rl_conv(x, f, k=1, s=1, p='same'):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(f, k, strides=s, padding=p)(x)
        return x
  
  
    def dense_block(tensor, r):
        for _ in range(r):
            x = bn_rl_conv(tensor, 4*f)
            x = bn_rl_conv(x, f, 3)
            tensor = Concatenate()([tensor, x])
        return tensor
  
  
    def transition_block(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x
  
  
    input = Input(img_shape)
  
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)
  
    for r in repetitions:
        d = dense_block(x, r)
        x = transition_block(d)
  
    x = GlobalAvgPool2D()(d)
  
    output = Dense(n_classes, activation='softmax')(x)
  
    model = Model(input, output)
    return model

def xception(input_shape, n_classes):
    
    def conv_bn(x, f, k, s=1, p='same'):
        x = Conv2D(f, k, strides=s, padding=p, use_bias=False)(x)
        x = BatchNormalization()(x)
        return x
    
    def sep_bn(x, f, k, s=1, p='same'):
        x = SeparableConv2D(f, k, strides=s, padding=p, use_bias=False)(x)
        x = BatchNormalization()(x)
        return x
    
    def entry_flow(x):
        x = conv_bn(x, 32, 3, 2)
        x = ReLU()(x)
        x = conv_bn(x, 64, 3)
        tensor = ReLU()(x)

        x = sep_bn(tensor, 128, 3)
        x = ReLU()(x)
        x = sep_bn(x, 128, 3)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        tensor = conv_bn(tensor, 128, 1, 2)

        x = Add()([tensor, x])
        x = ReLU()(x)
        x = sep_bn(x, 256, 3)
        x = ReLU()(x)
        x = sep_bn(x, 256, 3)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        tensor = conv_bn(tensor, 256, 1, 2)

        x = Add()([tensor, x])
        x = ReLU()(x)
        x = sep_bn(x, 728, 3)
        x = ReLU()(x)
        x = sep_bn(x, 728, 3)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        tensor = conv_bn(tensor, 728, 1, 2)
        x = Add()([tensor, x])

        return x
    
    def middle_flow(tensor):
        for _ in range(8):
            x = ReLU()(tensor)
            x = sep_bn(x, 728, 3)
            x = ReLU()(x)
            x = sep_bn(x, 728, 3)
            x = ReLU()(x)
            x = sep_bn(x, 728, 3)

            tensor = Add()([tensor, x])
    
        return tensor

    def exit_flow(tensor):
        x = ReLU()(tensor)
        x = sep_bn(x, 728, 3)
        x = ReLU()(x)
        x = sep_bn(x, 1024, 3)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        tensor = conv_bn(tensor, 1024, 1, 2)

        x = Add()([tensor, x])
        x = sep_bn(x, 1536, 3)
        x = ReLU()(x)
        x = sep_bn(x, 2048, 3)
        x = ReLU()(x)
        x = GlobalAvgPool2D()(x)
        x = Dense(n_classes, activation='softmax')(x)

        return x
    
    input = Input(input_shape)
    
    x = entry_flow(input)
    x = middle_flow(x)
    output = exit_flow(x)
    
    model = Model(input, output)
    
    return model

def yolo(input_shape=(448, 448, 3), n_outputs=30):
    activation = LeakyReLU(0.1)

    def conv_1_3(x, f1, f2, r=1):
        for _ in range(r):
            x = Conv2D(f1, 1, padding='same', activation=activation)(x)
            x = Conv2D(f2, 3, padding='same', activation=activation)(x)
        return x
    
    input = Input(input_shape)

    x = Conv2D(64, 7, strides=2, padding='same', activation=activation)(input)
    x = MaxPool2D(2, strides=2, padding='same')(x)

    x = Conv2D(192, 3, padding='same', activation=activation)(x)
    x = MaxPool2D(2, strides=2, padding='same')(x)

    x = conv_1_3(x, 128, 256)
    x = conv_1_3(x, 256, 512)
    x = MaxPool2D(2, strides=2, padding='same')(x)

    x = conv_1_3(x, 256, 512, 4)
    x = conv_1_3(x, 512, 1024)
    x = MaxPool2D(2, strides=2, padding='same')(x)

    x = conv_1_3(x, 512, 1024, 2)
    x = Conv2D(1024, 3, padding='same', activation=activation)(x)
    x = Conv2D(1024, 3, strides=2, padding='same', activation=activation)(x)

    x = Conv2D(1024, 3, padding='same', activation=activation)(x)
    x = Conv2D(1024, 3, padding='same', activation=activation)(x)

    x = Dense(4096, activation=activation)(x)
    output = Dense(n_outputs)(x)
    
    model = Model(input, output)
    return model