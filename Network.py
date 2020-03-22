import tensorflow as tf
import os
import tensorflow.keras as keras
import Model


class Network(object):
    def __init__(self, config, strategy=None):
        self.strategy = strategy
        self.Alpha = config.generatorAlpha
        self.G_weight_clip_value = config.G_clip_value
        self.D_weight_clip_value = config.D_clip_value
        self.batch_size = config.batch_size
        if strategy:
            self.global_batch_size = strategy.num_replicas_in_sync * self.batch_size
        model = Model.Model(config)
        self.G_x2y, self.G_x2y_Cycle = model.Generator()
        self.G_y2x, self.G_y2x_Cycle = model.Generator()
        self.D_X = model.Discriminator()
        self.D_Y = model.Discriminator()
        self.Generator_train_variable = self.getGenTrainVariable()
        self.Discriminator_train_variable = self.getDisTrainVariable()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

        self.summary_writer = tf.summary.create_file_writer(os.path.join(config.checkpoint_prefix, 'summary'))
        checkpoint = tf.train.Checkpoint(optimizer_g=self.generator_optimizer,
                                         optimizer_d=self.discriminator_optimizer,
                                         G_x2y=self.G_x2y, G_y2x=self.G_y2x,
                                         G_x2y_Cycle=self.G_x2y_Cycle, G_y2x_Cycle=self.G_y2x_Cycle,
                                         D_X=self.D_X, D_Y=self.D_Y)
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=config.checkpoint_prefix, max_to_keep=20)
        if self.ckpt_manager.latest_checkpoint:
            checkpoint.restore(self.ckpt_manager.latest_checkpoint)

    def average_loss(self, loss):
        # return tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)
        return loss

    def regularizer_loss(self, model):
        return tf.add_n(getattr(model, 'losses'))

    def get_regularizer_gen_loss(self):
        loss_Generator_regularizer = tf.nn.scale_regularization_loss(
            self.regularizer_loss(self.G_x2y) +
            self.regularizer_loss(self.G_y2x) +
            self.regularizer_loss(self.G_x2y_Cycle) +
            self.regularizer_loss(self.G_y2x_Cycle))
        return loss_Generator_regularizer

    def get_regularizer_dis_loss(self):

        loss_Discriminator_regularizer = tf.nn.scale_regularization_loss(
            self.regularizer_loss(self.D_X) +
            self.regularizer_loss(self.D_Y))
        return loss_Discriminator_regularizer

    def getGenTrainVariable(self):
        gen_x2y = self.G_x2y.trainable_variables
        gen_x2y_Cycle = self.G_x2y_Cycle.trainable_variables
        for v1, v2 in zip(gen_x2y, gen_x2y_Cycle):
            if not v1.name == v2.name:
                gen_x2y.append(v2)
        gen_y2x = self.G_y2x.trainable_variables
        gen_y2x_Cycle = self.G_y2x_Cycle.trainable_variables
        for v1, v2 in zip(gen_y2x, gen_y2x_Cycle):
            if not v1.name == v2.name:
                gen_y2x.append(v2)
        return gen_x2y + gen_y2x

    def getDisTrainVariable(self):
        return self.D_X.trainable_variables + self.D_Y.trainable_variables

    def strategy2Tensor(self, loss):
        # return tf.reduce_mean(self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None))
        return tf.reduce_mean(loss)

    def train_step_discriminator(self, inputs, label):
        with tf.GradientTape(persistent=True) as tape_Discriminator:
            '''生成'''
            fake_y = self.G_x2y(inputs)
            fake_x = self.G_y2x(label)
            '''鉴别'''
            dis_input = self.D_X(inputs)
            dis_label = self.D_Y(label)
            dis_fake_label = self.D_Y(fake_y)
            dis_fake_input = self.D_X(fake_x)
            '''loss'''
            loss_Discriminator_regularizer = self.get_regularizer_dis_loss()
            # loss_Discriminator = tf.losses.hinge(dis_input, dis_fake_input) + tf.losses.hinge(dis_label, dis_fake_label)
            loss_Discriminator = -tf.reduce_mean(dis_input) + tf.reduce_mean(dis_fake_input) - tf.reduce_mean(
                dis_label) + tf.reduce_mean(dis_fake_label)
            loss_Discriminator_total = self.average_loss(loss_Discriminator) + loss_Discriminator_regularizer
        discriminator_gradients = tape_Discriminator.gradient(loss_Discriminator_total,
                                                              self.Discriminator_train_variable)
        # discriminator_gradients = [tf.clip_by_value(grad, -self.D_weight_clip_value, self.D_weight_clip_value) for grad
        #                            in discriminator_gradients]

        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.Discriminator_train_variable))
        output_loss = [dis_input, dis_label, dis_fake_label, dis_fake_input, loss_Discriminator_regularizer,
                       loss_Discriminator, loss_Discriminator_total]
        return output_loss

    def train_step_generator(self, inputs, label):
        with tf.GradientTape(persistent=True) as tape_Generator:
            ''''''
            fake_y = self.G_x2y(inputs)
            cycle_x = self.G_y2x_Cycle(fake_y)
            fake_x = self.G_y2x(label)
            cycle_y = self.G_x2y_Cycle(fake_x)
            ''''''
            dis_fake_label_gen = self.D_Y(fake_y)
            dis_fake_input_gen = self.D_X(fake_x)
            '''loss'''
            loss_cycle = tf.losses.MSE(inputs, cycle_x) + tf.losses.MSE(label, cycle_y)
            loss_identity = tf.losses.MSE(inputs, fake_y) + tf.losses.MSE(label, fake_x)
            loss_Generator = -tf.reduce_mean(dis_fake_label_gen) - tf.reduce_mean(dis_fake_input_gen)
            loss_Generator += loss_cycle * (self.Alpha ** 2) + loss_identity * self.Alpha
            loss_Generator_regularizer = self.get_regularizer_gen_loss()
            loss_Generator_total = self.average_loss(loss_Generator) + loss_Generator_regularizer
        generator_gradients = tape_Generator.gradient(loss_Generator_total,
                                                      self.Generator_train_variable)
        # generator_gradients = [tf.clip_by_value(grad, -self.G_weight_clip_value, self.G_weight_clip_value) for grad in
        #                        generator_gradients]
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.Generator_train_variable))
        output_loss = [dis_fake_label_gen, dis_fake_input_gen, loss_cycle, loss_identity, loss_Generator,
                       loss_Generator_regularizer, loss_Generator_total]
        return output_loss

    @tf.function
    def distributed_train_step(self, inputs, labels, is_Gen_Train):
        # loss = self.strategy.experimental_run_v2(self.train_step_discriminator, args=(inputs, labels))
        loss = self.train_step_discriminator(inputs, labels)
        loss = [self.strategy2Tensor(lossTemp) for lossTemp in loss]
        loss_dict = {'dis_input': loss[0],
                     'dis_label': loss[1],
                     'dis_fake_label': loss[2],
                     'dis_fake_input': loss[3],
                     'loss_Discriminator_regularizer': loss[4],
                     'loss_Discriminator': loss[5],
                     'loss_Discriminator_total': loss[6]
                     }
        if is_Gen_Train:
            # loss = self.strategy.experimental_run_v2(self.train_step_generator, args=(inputs, labels))
            loss = self.train_step_generator(inputs, labels)
            loss = [self.strategy2Tensor(lossTemp) for lossTemp in loss]
            loss_dict['dis_fake_label_gen'] = loss[0]
            loss_dict['dis_fake_input_gen'] = loss[1]
            loss_dict['loss_cycle'] = loss[2]
            loss_dict['loss_identity'] = loss[3]
            loss_dict['loss_Generator'] = loss[4]
            loss_dict['loss_Generator_regularizer'] = loss[5]
            loss_dict['loss_Generator_total'] = loss[6]
        return loss_dict
