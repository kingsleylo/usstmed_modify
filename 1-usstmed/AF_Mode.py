'''
-*- coding: utf-8 -*-
@Author   : Jingluo
@FILE     : AF_Mode
@Time     : 2023/4/18 11:18
@Software : 生理信号挑战赛
@Email    : 1871513040@qq.com
@Github   : 
@ID       : PyCharm
@Desc     : 
'''
from tensorflow.keras.layers import Input, SeparableConv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, \
    GlobalAveragePooling1D, Dense, Reshape, Multiply, Add
from tensorflow.keras.models import Model


def AF_model(input_shape):
    input_layer = Input(shape=input_shape, name='input_layer')

    x = SeparableConv1D(filters=64, kernel_size=3, padding='same', name='separable_conv1d_79')(input_layer)
    x = BatchNormalization(name='batch_normalization_79')(x)
    x = Activation('relu', name='activation_67')(x)
    x = MaxPooling1D(pool_size=2, name='max_pooling1d_3')(x)

    x = SeparableConv1D(filters=64, kernel_size=3, padding='same', name='separable_conv1d_80')(x)
    x = BatchNormalization(name='batch_normalization_80')(x)
    x = Activation('relu', name='activation_68')(x)
    x = Dropout(rate=0.5, name='dropout_37')(x)

    x = SeparableConv1D(filters=64, kernel_size=3, padding='same', name='separable_conv1d_81')(x)
    x = BatchNormalization(name='batch_normalization_81')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_33')(x)
    x = Dense(units=4, activation='softmax', name='dense_69')(x)
    x = Dense(units=64, activation='relu', name='dense_70')(x)
    x = Reshape(target_shape=(1, 64), name='reshape_33')(x)
    x = Multiply(name='multiply_33')([x, BatchNormalization(name='batch_normalization_81')(x)])

    shortcut = MaxPooling1D(pool_size=2, name='max_pooling1d_3')(input_layer)
    shortcut = SeparableConv1D(filters=64, kernel_size=3, padding='same', name='separable_conv1d_82')(shortcut)
    shortcut = BatchNormalization(name='batch_normalization_82')(shortcut)
    shortcut = Activation('relu', name='activation_70')(shortcut)

    x = Add(name='add_33')([x, shortcut])
    x = Activation('relu', name='activation_69')(x)

    x = SeparableConv1D(filters=64, kernel_size=3, padding='same', name='separable_conv1d_83')(x)
    x = BatchNormalization(name='batch_normalization_83')(x)
    x = Activation('relu', name='activation_70')(x)
    x = Dropout(rate=0.5, name='dropout_38')(x)

    x = SeparableConv1D(filters=64, kernel_size=3, padding='same', name='separable_conv1d_84')(x)
    x = BatchNormalization(name='batch_normalization_84')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_34')(x)
    x = Dense(units=4, activation='softmax', name='dense_71')(x)
    x = Dense(units=64, activation='relu', name='dense_72')(x)
    x = Reshape(target_shape=(1, 64), name='reshape_34')(x)
    x = Multiply(name='multiply_34')([x, BatchNormalization(name='batch_normalization_83')(x)])
    x = Add(name='add_34')([x, Activation('relu', name='activation_69')(x)])
    x = Activation('relu', name='activation_71')(x)

    x = SeparableConv1D(filters=64, kernel_size=3, padding='same', name='separable_conv1d_84')(x)
    x = BatchNormalization(name='batch_normalization_84')(x)
    x = Activation('relu', name='activation_72')(x)

    # SeparableConv1D layer with name separable_conv1d_85
    x = SeparableConv1D(64, kernel_size=3, padding='same', name='separable_conv1d_85')(x)
    x = BatchNormalization(name='batch_normalization_85')(x)
    x = Activation('relu')(x)

    # GlobalAveragePooling1D layer with name global_average_pooling1d_35
    x = GlobalAveragePooling1D(name='global_average_pooling1d_35')(x)

    # Dense layers with names dense_73 and dense_74
    x = Dense(4, activation='softmax', name='dense_73')(x)
    x = Dense(64, activation='relu', name='dense_74')(x)

    # Reshape layer with name reshape_35
    x = Reshape((1, 64), name='reshape_35')(x)

    # SeparableConv1D layer with name separable_conv1d_86
    skip = SeparableConv1D(64, kernel_size=1, padding='same', name='separable_conv1d_86')(x)
    skip = BatchNormalization(name='batch_normalization_86')(skip)

    # Multiply layer with name multiply_35
    x = Multiply(name='multiply_35')([x, BatchNormalization()(skip)])

    # Add layer with name add_35
    x = Add(name='add_35')([x, skip])
    x = Activation('relu')(x)

    # SeparableConv1D layer with name separable_conv1d_87
    x = SeparableConv1D(128, kernel_size=3, padding='same', name='separable_conv1d_87')(x)
    x = BatchNormalization(name='batch_normalization_87')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5, name='dropout_40')(x)

    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='separable_conv1d_85')(x)
    x = Dropout(0.1, name='dropout_39')(x)
    x = BatchNormalization(name='batch_normalization_85')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_35')(x)
    x = Dense(4, activation='relu', name='dense_73')(x)
    x = Dense(64, activation='relu', name='dense_74')(x)
    x = Reshape((1, 64), name='reshape_35')(x)
    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='separable_conv1d_86')(inputs)
    x = Multiply(name='multiply_35')([x, BatchNormalization(name='batch_normalization_85')(x)])
    x = BatchNormalization(name='batch_normalization_86')(x)
    x = Add(name='add_35')(
        [x, Multiply(name='multiply_35.1')([x, BatchNormalization(name='batch_normalization_85.1')(x)])])
    x = Activation('relu', name='activation_73')(x)
    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='separable_conv1d_87')(x)
    x = BatchNormalization(name='batch_normalization_87')(x)
    x = Activation('relu', name='activation_74')(x)
    x = Dropout(0.1, name='dropout_40')(x)
    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='separable_conv1d_88')(x)
    x = BatchNormalization(name='batch_normalization_88')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_36')(x)
    x = Dense(8, activation='relu', name='dense_75')(x)
    x = Dense(128, activation='relu', name='dense_76')(x)
    x = Reshape((1, 128), name='reshape_36')(x)
    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='separable_conv1d_89')(x)
    x = Multiply(name='multiply_36')([x, BatchNormalization(name='batch_normalization_88.1')(x)])
    x = BatchNormalization(name='batch_normalization_89')(x)
    x = Add(name='add_36')(
        [x, Multiply(name='multiply_36.1')([x, BatchNormalization(name='batch_normalization_88.2')(x)])])
    x = Activation('relu', name='activation_75')(x)
    x = SeparableConv1D(128, 3, activation='relu', padding='same', name='separable_conv1d_90')(x)
    x = BatchNormalization(name='batch_normalization_90')(x)
    x = Activation('relu', name='activation_76')(x)
    x = Dropout(0.1, name='dropout_41')(x)
