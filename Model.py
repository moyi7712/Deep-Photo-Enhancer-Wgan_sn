import tensorflow as tf
import tensorflow.keras as keras
from Layer import Conv2D


class Model(object):
    def __init__(self, config):
        self.G_initializer_kernel = getattr(keras.initializers, config.G_initializer_kernel_type)(
            config.G_initializer_kernel_weight)
        self.G_initializer_bias = getattr(keras.initializers, config.G_initializer_bias_type)(
            config.G_initializer_bias_weight)

        self.G_regulazition_kernel = getattr(keras.regularizers, config.G_regulazition_kernel_type)(
            config.G_regulazition_kernel_weight)
        self.G_regulazition_bias = getattr(keras.regularizers, config.G_regulazition_bias_type)(
            config.G_regulazition_bias_weight)

        self.G_activation = config.G_activation
        self.G_is_sn = config.G_is_sn

        self.D_initializer_kernel = getattr(keras.initializers, config.D_initializer_kernel_type)(
            config.D_initializer_kernel_weight)
        self.D_initializer_bias = getattr(keras.initializers, config.D_initializer_bias_type)(
            config.D_initializer_bias_weight)

        self.D_regulazition_kernel = getattr(keras.regularizers, config.D_regulazition_kernel_type)(
            config.D_regulazition_kernel_weight)
        self.D_regulazition_bias = getattr(keras.regularizers, config.D_regulazition_bias_type)(
            config.D_regulazition_bias_weight)
        self.D_activation = config.D_activation
        self.D_is_sn = config.D_is_sn

    def Conv2D_G(self, filter_num, kernel_size, stride, name, is_upsample=False, activate=None, padding='SYMMETRIC'):
        layer = Conv2D(kernel_num=filter_num, kernel_size=kernel_size, stride=stride,
                       padding=padding, name=name, is_sn=self.G_is_sn,
                       activation=self.G_activation if not activate else activate if not activate == 'None' else None,
                       is_upsample=is_upsample,
                       kernel_initializer=self.G_initializer_kernel,
                       kernel_regularizer=self.G_regulazition_kernel,
                       bias_initializer=self.G_initializer_bias,
                       bias_regularizer=self.G_regulazition_bias)
        return layer

    def Conv2D_D(self, filter_num, kernel_size, stride, name, is_upsample=False, activate=None, padding='SYMMETRIC'):
        layer = Conv2D(kernel_num=filter_num, kernel_size=kernel_size, stride=stride,
                       padding=padding, name=name, is_sn=self.D_is_sn,
                       activation=self.D_activation if not activate else activate if not activate == 'None' else None,
                       is_upsample=is_upsample,
                       kernel_initializer=self.D_initializer_kernel,
                       kernel_regularizer=self.D_regulazition_kernel,
                       bias_initializer=self.D_initializer_bias,
                       bias_regularizer=self.D_regulazition_bias)
        return layer

    def BN_G(self, name):
        return keras.layers.BatchNormalization(name=name, epsilon=1e-5, momentum=0.99)

    def BN_D(self, name):
        return keras.layers.BatchNormalization(name=name, epsilon=1e-5, momentum=0.99)

    def Generator(self):
        ConvStack = self.GeneratorConvStack()
        return self.CreatGenerator(ConvStack, flage=None), self.CreatGenerator(ConvStack, flage='Cycle')

    def GeneratorConvStack(self):
        downsample_Conv_Stack = [
            self.Conv2D_G(16, 3, 1, padding='SYMMETRIC', name='conv_expand_channel'),
            self.Conv2D_G(32, 5, 2, padding='SYMMETRIC', name='conv_downsample_1'),
            self.Conv2D_G(64, 5, 2, padding='SYMMETRIC', name='conv_downsample_2'),
            self.Conv2D_G(128, 5, 2, padding='SYMMETRIC', name='conv_downsample_3'),
            self.Conv2D_G(128, 5, 2, padding='SYMMETRIC', name='conv_downsample_4'),
        ]

        global_Conv_Stack = [
            self.Conv2D_G(128, 5, 2, padding='SYMMETRIC', name='conv_global_1'),
            self.Conv2D_G(128, 5, 2, padding='SYMMETRIC', name='conv_global_2'),
            self.Conv2D_G(128, 8, 1, padding='valid', name='conv_global_3'),
            self.Conv2D_G(128, 1, 1, padding='valid', name='conv_global_4', activate='None'),
        ]

        featureFusion_Conv_Stack = [
            self.Conv2D_G(128, 3, 1, padding='SYMMETRIC', name='conv_feature_1', activate='None'),
            self.Conv2D_G(128, 1, 1, padding='SYMMETRIC', name='conv_feature_2'),
        ]
        upsample_Conv_Stack = [
            self.Conv2D_G(128, 3, 1, padding='SYMMETRIC', name='conv_upsample_1', is_upsample=True),
            self.Conv2D_G(128, 3, 1, padding='SYMMETRIC', name='conv_upsample_2', is_upsample=True),
            self.Conv2D_G(64, 3, 1, padding='SYMMETRIC', name='conv_upsample_3', is_upsample=True),
            self.Conv2D_G(32, 3, 1, padding='SYMMETRIC', name='conv_upsample_4', is_upsample=True),
            self.Conv2D_G(16, 3, 1, padding='SYMMETRIC', name='conv_upsample_5'),
            self.Conv2D_G(3, 3, 1, padding='SYMMETRIC', name='conv_upsample_6', activate='None'),
        ]
        return (downsample_Conv_Stack, global_Conv_Stack, featureFusion_Conv_Stack, upsample_Conv_Stack)

    def GeneratorNormStack(self, flage=None):
        flage = '_'+flage if flage else ''
        downsample_Norm_Stack = [
            self.BN_G('bn_downsample_1' + flage),
            self.BN_G('bn_downsample_2' + flage),
            self.BN_G('bn_downsample_3' + flage),
            self.BN_G('bn_downsample_4' + flage),
            self.BN_G('bn_downsample_5' + flage)
        ]
        global_Norm_Stack = [
            self.BN_G('bn_global_1' + flage),
            self.BN_G('bn_global_2' + flage),
            None,
            None
        ]

        featureFusion_Norm_Stack = [
            None,
            self.BN_G('bn_feature_2' + flage)
        ]
        upsample_Norm_Stack = [
            self.BN_G('bn_upsample_1' + flage),
            self.BN_G('bn_upsample_2' + flage),
            self.BN_G('bn_upsample_3' + flage),
            self.BN_G('bn_upsample_4' + flage),
            self.BN_G('bn_upsample_5' + flage),
            None
        ]
        return downsample_Norm_Stack, global_Norm_Stack, featureFusion_Norm_Stack, upsample_Norm_Stack

    def CreatGenerator(self, ConvStack, flage):
        downsample_Conv_Stack, global_Conv_Stack, featureFusion_Conv_Stack, upsample_Conv_Stack = ConvStack
        downsample_Norm_Stack, global_Norm_Stack, featureFusion_Norm_Stack, upsample_Norm_Stack = self.GeneratorNormStack(
            flage=flage)
        inputs = keras.layers.Input([512, 512, 3])
        down_concat = []
        down_ = inputs
        for c, n in zip(downsample_Conv_Stack, downsample_Norm_Stack):
            down_ = n(c(down_))
            down_concat.append(down_)
        global_ = down_
        for c, n in zip(global_Conv_Stack, global_Norm_Stack):
            global_ = c(global_)
            if n:
                global_ = n(global_)
        global_ = tf.tile(global_, [1, 32, 32, 1], name='tile')

        featureFusion_ = down_
        for c, n, i in zip(featureFusion_Conv_Stack, featureFusion_Norm_Stack, [True, None]):
            featureFusion_ = c(featureFusion_)
            if n:
                featureFusion_ = n(featureFusion_)
            if i:
                featureFusion_ = tf.concat(values=[featureFusion_, global_], axis=3, name='FeatureConcat')

        up_ = featureFusion_
        for c, n, i in zip(upsample_Conv_Stack, upsample_Norm_Stack, [4, 3, 2, 1, None, None]):
            up_ = c(up_)
            name = c.name[-1]
            if i:
                up_ = tf.concat(values=[up_, down_concat[i - 1]], axis=3, name='upsample_concat_' + name)
            if n:
                up_ = n(up_)
        # outputs = up_
        outputs = tf.clip_by_value(up_ + inputs, 0, 1)
        return keras.Model(inputs=inputs, outputs=outputs)

    def DiscriminatorStack(self):
        Stack = [
            self.Conv2D_D(16, 3, 1, padding='SYMMETRIC', name='expand_channel'),
            self.BN_D(name='expand_channel_norm'),
            self.Conv2D_D(32, 5, 2, padding='SYMMETRIC', name='downsample_conv_1'),
            self.BN_D(name='downsample_norm_1'),
            self.Conv2D_D(64, 5, 2, padding='SYMMETRIC', name='downsample_conv_2'),
            self.BN_D(name='downsample_norm_2'),
            self.Conv2D_D(128, 5, 2, padding='SYMMETRIC', name='downsample_conv_3'),
            self.BN_D(name='downsample_norm_3'),
            self.Conv2D_D(128, 5, 2, padding='SYMMETRIC', name='downsample_conv_4'),
            self.BN_D(name='downsample_norm_4'),
            self.Conv2D_D(128, 5, 2, padding='SYMMETRIC', name='downsample_conv_5'),
            self.BN_D(name='downsample_norm_5'),
            self.Conv2D_D(1, 5, stride=1, padding='CONSTANT', name='downsample_conv_6'),
            self.BN_D(name='downsample_norm_6'),
            self.Conv2D_D(1, 16, stride=1, padding='valid', name='downsample_conv_7'),



        ]
        return Stack

    def Discriminator(self):
        Stack = self.DiscriminatorStack()
        inputs = keras.layers.Input(shape=[512, 512, 3])
        outputs = inputs
        for l in Stack:
            outputs = l(outputs)
        # outputs = tf.nn.sigmoid(outputs)
        outputs = tf.reduce_mean(outputs, axis=[1, 2, 3])

        return keras.Model(inputs=inputs, outputs=outputs)