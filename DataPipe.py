import tensorflow as tf
import os, json



class PipeLine(object):
    def __init__(self, config, batch_size=None):
        self.config = config
        self.batch_size = batch_size if batch_size else config.batch_size
        with open(config.filelist, 'r') as f:
            self.filelist = json.load(f)

    @tf.function
    def _imread(self, file_path):
        image = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(file_path)), tf.dtypes.float32)
        return image

    @tf.function
    def _preprocess_test(self, filename_input, filename_label):
        input = self._imread(filename_input)
        shape = tf.shape(input)
        shape = tf.cast(512*shape[:-1]/tf.reduce_max(shape), dtype=tf.int32)
        label = self._imread(filename_label)
        input = tf.image.resize(input, shape)
        label = tf.image.resize(label, shape)
        padding = tf.concat(
            [tf.constant([[0], [0], [0]]), tf.reshape(tf.constant([512, 512, 3]) - tf.shape(input), shape=(3, 1))],
            axis=1)
        if shape[0] > shape[1]:
            mask = tf.concat([tf.ones(shape), tf.zeros((512, 512-shape[1]))], axis=1)
        else:
            mask = tf.concat([tf.ones(shape), tf.zeros((512 - shape[0], 512))], axis=0)

        input = tf.pad(input, padding, mode='SYMMETRIC')
        label = tf.pad(label, padding, mode='SYMMETRIC')
        return input, label, tf.cast(mask, dtype=tf.bool)

    def _preprocess_train(self, filename_input, filename_label):
        input = self._imread(filename_input)
        shape_input = tf.shape(input)[:-1]
        shape_input = tf.cast(512 * shape_input / tf.reduce_min(shape_input), dtype=tf.int32)
        label = self._imread(filename_label)
        shape_label = tf.shape(label)[:-1]
        shape_label = tf.cast(512 * shape_label / tf.reduce_min(shape_label), dtype=tf.int32)
        input = tf.image.resize(input, shape_input)
        label = tf.image.resize(label, shape_label)
        input = tf.image.resize_with_crop_or_pad(input, 512, 512)
        label = tf.image.resize_with_crop_or_pad(label, 512, 512)
        return input, label

    def _dataset_test_build(self, inputs_list, lables_list):
        dataset = tf.data.Dataset.from_tensor_slices((inputs_list, lables_list))
        dataset = dataset.repeat(-1)
        dataset = dataset.map(self._preprocess_test)
        dataset = dataset.batch(2).shuffle(self.config.shuffle)
        return dataset

    def _dataset_train_build(self, inputs_list, lables_list):
        dataset = tf.data.Dataset.from_tensor_slices((inputs_list, lables_list))
        dataset = dataset.repeat(-1)
        dataset = dataset.map(self._preprocess_train, num_parallel_calls=self.config.num_parallel_calls)
        dataset = dataset.batch(self.batch_size) \
            .prefetch(buffer_size=self.config.buffer_size)\
            .shuffle(self.config.shuffle)

        return dataset

    def test(self):
        inputs_list = [os.path.join(self.config.input_path, temp) for temp in self.filelist['test']]
        labels_list = [os.path.join(self.config.label_path, temp) for temp in self.filelist['test']]
        return self._dataset_test_build(inputs_list, labels_list)

    def train(self):
        inputs_list = [os.path.join(self.config.input_path, temp) for temp in self.filelist['input']]
        if self.config.is_pair:
            labels_list = [os.path.join(self.config.label_path, temp) for temp in self.filelist['input']]
        else:
            labels_list = [os.path.join(self.config.label_path, temp) for temp in self.filelist['label']]
        return self._dataset_train_build(inputs_list, labels_list)

