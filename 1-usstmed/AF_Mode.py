'''
-*- coding: utf-8 -*-
@Author   : Jingluo
@FILE     : AF_Mode
@Time     : 2023/4/18 11:18
@project  : 生理信号挑战赛
@Email    : 1871513040@qq.com
@Github   : 
@IDE      : PyCharm
@Desc     : This project was created at Hubei University of Technology
'''
from tensorflow.keras.layers import Input, SeparableConv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, \
    GlobalAveragePooling1D, Dense, Reshape, Multiply, Add
from tensorflow.keras.models import Model


def AF_model(input_shape):
    '''
    Deep Resnet！！！
    '''
    inputs = Input(input_shape, name='input_4')
    x = SeparableConv1D(64, 3, padding='same', name='separable_conv1d_79')(inputs)
    x = BatchNormalization(name='batch_normalization_79')(x)
    x = Activation('relu', name='activation_67')(x)
    x1 = MaxPooling1D(pool_size=3, strides=2, padding='same', name='max_pooling1d_3')(x)
    x = SeparableConv1D(64, 3, padding='same', name='separable_conv1d_80')(x1)
    x = BatchNormalization(name='batch_normalization_80')(x)
    x = Activation('relu', name='activation_68')(x)
    x = Dropout(0.1, name='dropout_37')(x)
    x = SeparableConv1D(64, 3, padding='same', name='separable_conv1d_81')(x)
    x2 = BatchNormalization(name='batch_normalization_81')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_33')(x2)
    x = Dense(4, activation='softmax', name='dense_69')(x)
    x = Dense(64, name='dense_70')(x)
    x = Reshape((1, 64), name='reshape_33')(x)
    x = Multiply(name='multiply_33')([x, x2])
    x = Add(name='add_33')([x, x1])
    x3 = Activation('relu', name='activation_69')(x)
    x = SeparableConv1D(64, 3, padding='same', name='separable_conv1d_82')(x3)
    x = BatchNormalization(name='batch_normalization_82')(x)
    x = Activation('relu', name='activation_70')(x)
    x = Dropout(0.1, name='dropout_38')(x)
    x = SeparableConv1D(64, 3, padding='same', name='separable_conv1d_83')(x)
    x4 = BatchNormalization(name='batch_normalization_83')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_34')(x4)
    x = Dense(4, activation='softmax', name='dense_71')(x)
    x = Dense(64, name='dense_72')(x)
    x = Reshape((1, 64), name='reshape_34')(x)
    x = Multiply(name='multiply_34')([x, x4])
    x = Add(name='add_34')([x,x3])
    x = Activation('relu', name='activation_71')(x)
    x = SeparableConv1D(64, 3, padding='same', name='separable_conv1d_84')(x)
    x = BatchNormalization(name='batch_normalization_84')(x)
    x = Activation('relu', name='activation_72')(x)
    x = Dropout(rate=0.1, name='dropout_39')(x)

    x = SeparableConv1D(filters=64, kernel_size=3, padding='same', name='separable_conv1d_85')(x)
    x5 = BatchNormalization(name='batch_normalization_85')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_35')(x5)
    x = Dense(units=4, activation='softmax', name='dense_73')(x)
    x = Dense(units=64, activation='relu', name='dense_74')(x)
    x = Reshape(target_shape=(1, 64), name='reshape_35')(x)
    x = SeparableConv1D(filters=64, kernel_size=3, padding='same', name='separable_conv1d_86')(x)
    x = Multiply(name='multiply_35')([x, x5])
    x6 = BatchNormalization(name='batch_normalization_86')(x)
    x = Add(name='add_35')([x, x6])
    x9 = Activation('relu', name='activation_73')(x)
    x = SeparableConv1D(filters=128, kernel_size=3, padding='same', name='separable_conv1d_87')(x9)
    x = BatchNormalization(name='batch_normalization_87')(x)
    x = Activation('relu', name='activation_74')(x)
    x = Dropout(rate=0.2, name='dropout_40')(x)
    x = SeparableConv1D(filters=128, kernel_size=3, padding='same', name='separable_conv1d_88')(x)
    x7 = BatchNormalization(name='batch_normalization_88')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_36')(x7)
    x = Dense(units=8, activation='softmax', name='dense_75')(x)
    x = Dense(units=128, activation='relu', name='dense_76')(x)
    x8 = Reshape(target_shape=(1, 128), name='reshape_36')(x)
    x10 = SeparableConv1D(filters=128, kernel_size=3, padding='same', name='separable_conv1d_89')(x9)
    x11 = Multiply(name='multiply_36')(
        [x7, x8])
    x = BatchNormalization(name='batch_normalization_89')(x10)
    x = Add(name='add_36')([x11, x])

    x12 = Activation('relu', name='activation_75')(x)
    x = SeparableConv1D(128, kernel_size=3, padding='same', name='separable_conv1d_90')(x)
    x = BatchNormalization(name='batch_normalization_90')(x)
    x = Activation('relu', name='activation_76')(x)
    x = Dropout(0.1, name='dropout_41')(x)
    x = SeparableConv1D(128, kernel_size=3, padding='same', name='separable_conv1d_91')(x)
    x13 = BatchNormalization(name='batch_normalization_91')(x)

    gap = GlobalAveragePooling1D(name='global_average_pooling1d_37')(x13)

    x = Dense(8, activation='softmax', name='dense_77')(gap)
    x = Dense(128, activation='relu', name='dense_78')(x)
    x = Reshape((1, 128), name='reshape_37')(x)
    x = Multiply(name='multiply_37')([x, x13])

    x = Add(name='add_37')([x, x12])
    x14 = Activation('relu', name='activation_77')(x)
    x = SeparableConv1D(128, kernel_size=3, padding='same', name='separable_conv1d_92')(x14)
    x = BatchNormalization(name='batch_normalization_92')(x)
    x = Activation('relu', name='activation_78')(x)
    x = Dropout(0.1, name='dropout_42')(x)
    x = SeparableConv1D(128, kernel_size=3, padding='same', name='separable_conv1d_93')(x)
    x15 = BatchNormalization(name='batch_normalization_93')(x)

    gap = GlobalAveragePooling1D(name='global_average_pooling1d_38')(x15)

    x = Dense(8, activation='softmax', name='dense_79')(gap)
    x = Dense(128, activation='relu', name='dense_80')(x)
    x = Reshape((1, 128), name='reshape_38')(x)
    x = Multiply(name='multiply_38')([x, x15])

    x = Add(name='add_38')([x, x14])
    x16 = Activation('relu', name='activation_79')(x)
    x = SeparableConv1D(128, kernel_size=3, padding='same', name='separable_conv1d_94')(x16)
    x = BatchNormalization(name='batch_normalization_94')(x)
    x = Activation('relu', name='activation_80')(x)
    x = Dropout(0.1, name='dropout_43')(x)

    x = SeparableConv1D(128, kernel_size=3, padding='same', name='separable_conv1d_95')(x)
    x17 = BatchNormalization(name='batch_normalization_95')(x)
    x22 = Activation('relu',name="activation_81")(x17)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_39')(x22)

    # First Dense Layer
    x = Dense(8, activation='softmax', name='dense_81')(x)
    x = Dense(128, activation='relu', name='dense_82')(x)
    x18 = Reshape((1, 128), name='reshape_39')(x)

    # Second Separable Convolution Layer
    x19 = SeparableConv1D(128, kernel_size=3, padding='same', name='separable_conv1d_96')(x16)
    x20 = Multiply(name='multiply_39')([x17, x18])
    x = BatchNormalization(name='batch_normalization_96')(x19)
    x = Add(name='add_39')([x, x20])
    x = Activation('relu')(x)

    # Third Separable Convolution Layer
    x = SeparableConv1D(256, kernel_size=3, padding='same', name='separable_conv1d_97')(x)
    x = BatchNormalization(name='batch_normalization_97')(x)
    x = Activation('relu', name='activation_82')(x)
    x = Dropout(0.1, name='dropout_44')(x)

    # Fourth Separable Convolution Layer
    x = SeparableConv1D(256, kernel_size=3, padding='same', name='separable_conv1d_98')(x)
    x20 = BatchNormalization(name='batch_normalization_98')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_40')(x20)

    # Second Dense Layer
    x = Dense(16, activation='softmax', name='dense_83')(x)
    x = Dense(256, activation='relu', name='dense_84')(x)
    x21 = Reshape((1, 256), name='reshape_40')(x)

    # Fifth Separable Convolution Layer
    x23 = SeparableConv1D(256, kernel_size=3, padding='same', name='separable_conv1d_99')(x22)
    x = Multiply(name='multiply_40')([x21, x20])
    x = BatchNormalization(name='batch_normalization_99')(x23)
    x = Add(name='add_40')([x, x21])
    x24 = Activation('relu', name='activation_83')(x)

    # Sixth Separable Convolution Layer
    x = SeparableConv1D(256, kernel_size=3, padding='same', name='separable_conv1d_100')(x24)
    x = BatchNormalization(name='batch_normalization_100')(x)
    x = Activation('relu', name='activation_84')(x)
    x = Dropout(0.1, name='dropout_45')(x)
    x = SeparableConv1D(filters=256, kernel_size=3, padding='same', name='separable_conv1d_101')(x)
    x25 = BatchNormalization(name='batch_normalization_101')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_41')(x25)
    x = Dense(units=16, name='dense_85')(x)
    x = Dense(units=256, name='dense_86')(x)
    x = Reshape((1, 256), name='reshape_41')(x)
    x = Multiply(name='multiply_41')([x, x25])
    x = Add(name='add_41')([x, x24])
    x26 = Activation('relu', name='activation_85')(x)
    x = Dropout(0.1)(x26)

    x = SeparableConv1D(filters=256, kernel_size=3, padding='same', name='separable_conv1d_102')(x)
    x = BatchNormalization(name='batch_normalization_102')(x)
    x = Activation('relu', name='activation_86')(x)
    x = Dropout(0.1, name='dropout_47')(x)

    # 添加
    x = SeparableConv1D(256, 3, padding='same', name='separable_conv1d_103')(x)
    x27 = BatchNormalization(name='batch_normalization_103')(x)
    x = Activation('relu')(x27)
    x = Dropout(0.1, name='dropout_46')(x)

    # Global average pooling and fully connected layers
    x = GlobalAveragePooling1D(name='global_average_pooling1d_42')(x)
    x = Dense(16, activation='relu', name='dense_87')(x)
    x = Dense(256, activation='relu', name='dense_88')(x)
    x = Reshape((1, 256), name='reshape_42')(x)
    x = Multiply(name='multiply_42')([x,x27])
    x = Add(name='add_42')([x, x26])
    x28 = Activation('relu', name='activation_87')(x)

    # Second separable convolutional layer
    x = SeparableConv1D(256, 3, padding='same', name='separable_conv1d_104')(x28)
    x = BatchNormalization(name='batch_normalization_104')(x)
    x = Activation('relu', name='activation_88')(x)
    x = Dropout(0.1)(x)
    x = SeparableConv1D(256, 3, padding='same', name='separable_conv1d_105')(x)
    x29 = BatchNormalization(name='batch_normalization_105')(x)

    # Global average pooling and fully connected layers
    x = GlobalAveragePooling1D(name='global_average_pooling1d_43')(x29)
    x = Dense(16, activation='relu', name='dense_89')(x)
    x = Dense(256, activation='relu', name='dense_90')(x)
    x = Reshape((1, 256), name='reshape_43')(x)
    x = Multiply(name='multiply_43')([x, x29])
    x = Add(name='add_43')([x,x28])
    x30 = Activation('relu', name='activation_89')(x)

    # Third separable convolutional layer
    x = SeparableConv1D(256, 3, padding='same', name='separable_conv1d_106')(x30)
    x = BatchNormalization(name='batch_normalization_106')(x)
    x = Activation('relu', name='activation_90')(x)
    x = Dropout(0.1, name='dropout_48')(x)
    x = SeparableConv1D(256, 3, padding='same', name='separable_conv1d_107')(x)
    x29 = BatchNormalization(name='batch_normalization_107')(x)

    # Global average pooling and fully connected layers
    x = GlobalAveragePooling1D(name='global_average_pooling1d_44')(x29)
    x = Dense(16, activation='relu', name='dense_91')(x)
    x = Dense(256, activation='relu', name='dense_92')(x)
    x = Reshape((1, 256), name='reshape_44')(x)
    x = Multiply(name='multiply_44')([x, x29])
    x = Add(name='add_44')([x, x30])
    x = Activation('relu', name='activation_91')(x)



    # 新添加

    x = SeparableConv1D(256, 3, padding='same', name='separable_conv1d_108')(x)
    x = BatchNormalization(name='batch_normalization_108')(x)
    x = Activation('relu', name='activation_92')(x)
    x = Dropout(0.1, name='dropout_49')(x)
    x = SeparableConv1D(256, 3, padding='same', name='separable_conv1d_109')(x)
    x = BatchNormalization(name='batch_normalization_109')(x)
    x1 = GlobalAveragePooling1D(name='global_average_pooling1d_45')(x)
    x1 = Dense(16, activation='relu', name='dense_93')(x1)
    x1 = Dense(256, activation='relu', name='dense_94')(x1)
    x1 = Reshape((1, 256), name='reshape_45')(x1)
    x2 = SeparableConv1D(256, 3, padding='same', name='separable_conv1d_110')(x)
    x2 = Multiply(name='multiply_45')([x1, x2])
    x2 = BatchNormalization(name='batch_normalization_110')(x2)
    x = Add(name='add_45')([x, x2])
    x = Activation('relu', name='activation_93')(x)
    # Block 2
    x = SeparableConv1D(512, 3, padding='same', name='separable_conv1d_111')(x)
    x = BatchNormalization(name='batch_normalization_111')(x)
    x = Activation('relu', name='activation_94')(x)
    x = Dropout(0.1, name='dropout_50')(x)
    x = SeparableConv1D(512, 3, padding='same', name='separable_conv1d_112')(x)
    x = BatchNormalization(name='batch_normalization_112')(x)
    x1 = GlobalAveragePooling1D(name='global_average_pooling1d_46')(x)
    x1 = Dense(32, activation='relu', name='dense_95')(x1)
    x1 = Dense(512, activation='relu', name='dense_96')(x1)
    x1 = Reshape((1, 512), name='reshape_46')(x1)
    x2 = SeparableConv1D(512, 3, padding='same', name='separable_conv1d_113')(x)
    x2 = Multiply(name='multiply_46')([x1, x2])
    x2 = BatchNormalization(name='batch_normalization_113')(x2)
    x = Add(name='add_46')([x, x2])
    x31 = Activation('relu', name='activation_95')(x)
    #新添加2
    x = SeparableConv1D(filters=512, kernel_size=3, padding='same', name='separable_conv1d_114')(x31)
    x = BatchNormalization(name='batch_normalization_114')(x)
    x = Activation('relu', name='activation_96')(x)
    x = Dropout(rate=0.5, name='dropout_51')(x)

    # Separable Conv1D layer 2
    x = SeparableConv1D(filters=512, kernel_size=3, padding='same', name='separable_conv1d_115')(x)
    x32 = BatchNormalization(name='batch_normalization_115')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_47')(x32)

    # Dense layer 1
    x = Dense(units=32, activation='relu', name='dense_97')(x)
    x = Dense(units=512, activation='relu', name='dense_98')(x)
    x = Reshape(target_shape=(1, 512), name='reshape_47')(x)
    x = Multiply(name='multiply_47')([x, x32])
    x = Add(name='add_47')([x, x31])
    x32 = Activation('relu', name='activation_97')(x)

    # Separable Conv1D layer 3
    x = SeparableConv1D(filters=512, kernel_size=3, padding='same', name='separable_conv1d_116')(x32)
    x = BatchNormalization(name='batch_normalization_116')(x)
    x = Activation('relu', name='activation_98')(x)
    x = Dropout(rate=0.5, name='dropout_52')(x)

    # Separable Conv1D layer 4
    x = SeparableConv1D(filters=512, kernel_size=3, padding='same', name='separable_conv1d_117')(x)
    x33 = BatchNormalization(name='batch_normalization_117')(x)
    x = GlobalAveragePooling1D(name='global_average_pooling1d_48')(x33)

    # Dense layer 2
    x = Dense(units=32, activation='relu', name='dense_99')(x)
    x = Dense(units=512, activation='relu', name='dense_100')(x)
    x = Reshape(target_shape=(1, 512), name='reshape_48')(x)
    x = Multiply(name='multiply_48')([x, x33])
    x = Add(name='add_48')([x, x32])
    x = Activation('relu', name='activation_99')(x)
    x = Dropout(rate=0.5, name='dropout_53')(x)
    x = Dense(256, name='dense_101')(x)
    x = Dropout(0.5,name='dropout_54')(x)
    x = Dense(1, name='dense_102')(x)
    model = Model(inputs=inputs, outputs=x)
    return model



if __name__=='__main__':
    model=AF_model(input_shape=(6000, 2))
    model.summary()


