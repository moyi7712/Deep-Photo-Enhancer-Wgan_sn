import tensorflow as tf
import tensorflow.keras as keras


class SpectralNormalization(keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 # aggregation=tf.VariableAggregation.MEAN,
                                 name='sn_v',
                                 dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 # aggregation=tf.VariableAggregation.MEAN,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs, **kwargs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        # tf.distribute.StrategyExtended.update(self.layer.kernel, self.w / sigma)
        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        # tf.distribute.StrategyExtended.update(self.layer.kernel, self.w)

        self.layer.kernel.assign(self.w)


class Conv2D(keras.layers.Layer):
    def __init__(self, kernel_num, kernel_size, stride, name,
                 padding='valid',
                 kernel_initializer=None,
                 bias_initializer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activation=None,
                 is_sn=False,
                 is_upsample=False,
                 **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.padding = padding
        self.is_upsample = is_upsample
        self.activation = activation

        self.conv = keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=(stride, stride),
                                        padding='valid',
                                        kernel_initializer=keras.initializers.get(kernel_initializer),
                                        bias_initializer=keras.initializers.get(bias_initializer),

                                        bias_regularizer=keras.regularizers.get(bias_regularizer),
                                        kernel_regularizer=keras.regularizers.get(kernel_regularizer))
        if is_sn:
            self.conv = SpectralNormalization(self.conv)
        self.pad_size = (kernel_size - 1) // 2
        self._name = name

    def call(self, inputs, **kwargs):
        if not self.padding == 'valid':
            inputs = tf.pad(inputs,
                            [[0, 0], [self.pad_size, self.pad_size], [self.pad_size, self.pad_size], [0, 0]],
                            self.padding)

        output = self.conv(inputs=inputs)
        if self.is_upsample:
            shape = tf.cast(tf.shape(output), dtype=tf.int32)
            output = tf.image.resize(output, 2*shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if self.activation:
            act = getattr(tf.nn, self.activation)
            output = act(output)

        return output

