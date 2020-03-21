import yaml
import os
import tensorflow as tf
from Config import Config
from DataPipe import PipeLine
from Network import Network
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml', type=str, help='The train config file', required=False)
args = parser.parse_args()
config = Config(args.config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
net = Network(config.HyperPara())
datapipe = PipeLine(config.Dataset())#, batch_size=net.global_batch_size)
# train_data = strategy.experimental_distribute_dataset(datapipe.train())
train_data = datapipe.train()
test_dataset = datapipe.test().as_numpy_iterator()
# with strategy.scope():
step = 0

for inputs, labels in train_data:
    isgenTrain = True if step % 50 == 0 else False
    loss_dict = net.distributed_train_step(inputs, labels, isgenTrain)
    with net.summary_writer.as_default():
        for key in loss_dict.keys():
            tf.summary.scalar(key, loss_dict[key], step=step)
            loss_dict[key] = loss_dict[key].numpy()
    if isgenTrain:

        net.ckpt_manager.save()
        test_data = test_dataset.next()
        test_out = net.G_x2y(test_data[0])

        psnr = tf.image.psnr(tf.reduce_mean(test_data[0], axis=0), tf.reduce_mean(test_out, axis=0), max_val=1.0)
        ssim = tf.image.ssim(tf.reduce_mean(test_data[0], axis=0), tf.reduce_mean(test_out, axis=0), max_val=1.0)
        with net.summary_writer.as_default():
            tf.summary.image(name='input', data=test_data[0], step=step)
            tf.summary.image(name='output', data=test_out, step=step)
            tf.summary.image(name='lable', data=test_data[1], step=step)
            tf.summary.scalar('psnr', psnr, step=step)
            tf.summary.scalar('ssim', ssim, step=step)
        print('INFO: loss_G:{}, loss_D:{}'.format(loss_dict['loss_Generator_total'], loss_dict['loss_Discriminator_total']))
    step += 1