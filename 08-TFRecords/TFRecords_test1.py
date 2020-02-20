import os
import csv

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def to_tfrecord(in_file, out_file, prebatch):
    # feature_map = pd.read_csv('feature_map.csv', header=None)
    # sparse_size = feature_map[feature_map[2] == 'sparse'].shape[0]

    # options = tf.io.TFRecordOptions(compression_type='GZIP')
    writer = tf.io.TFRecordWriter(out_file)
    data = pd.read_csv(in_file).values
    print('Writing data to .tfrecord format...')
    deviceid, app_version, newsid_count, pos_deviceid_count_ratio, label = (list() for i in range(5))
    for num, line in enumerate(data):
        # is_test_file = (len(line) != feature_map.shape[0])
        deviceid.extend([int(line[0])])
        app_version.extend([int(line[1])])
        newsid_count.extend([int(line[2])])
        pos_deviceid_count_ratio.extend([float(line[3])])
        label.extend([int(line[4])])

        if ((num + 1) % prebatch) == 0:
            write_tf_example(writer, deviceid, app_version, newsid_count, pos_deviceid_count_ratio, label)
            deviceid, app_version, newsid_count, pos_deviceid_count_ratio, label = (list() for i in range(5))
        if ((num + 1) % (1024 * prebatch)) == 0:
            print('write %d lines' % num)


def write_tf_example(writer, deviceid, app_version, newsid_count, pos_deviceid_count_ratio, label):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'deviceid': _int64_feature(deviceid),
                'app_version': _int64_feature(app_version),
                'newsid_count': _int64_feature(newsid_count),
                'pos_deviceid_count_ratio': _float_feature(pos_deviceid_count_ratio),
                'label': _int64_feature(label),
            }))
    writer.write(example.SerializeToString())


def get_summary(file_path='/Users/vincent/Documents/projects/Competition/turingtopia/feature_map.csv'):
    with open(os.path.join(os.path.dirname(__file__),
                           file_path)) as f:
        feature_list = list(csv.reader(f))

    category_features = {item[1]: int(item[0]) for item in feature_list if item[2] == 'sparse'}
    numerical_features = {item[1]: int(item[0]) for item in feature_list if item[2] == 'dense'}
    voc_size = {item[1]: int(item[0]) for item in feature_list if item[2] == 'sparse'}
    return category_features, numerical_features, voc_size


def parser(record):
    feature_map = {'label': tf.io.FixedLenFeature([], tf.int64)}
    # for key, dim in SPARSE_FEATURES.items():
    #     #     feature_map[key] = tf.io.VarLenFeature([], tf.int64)
    for key, dim in CATEGORY_FEATURES.items():
        feature_map[key] = tf.io.FixedLenFeature([], tf.int64)
    for key, dim in NUMERICAL_FEATURES.items():
        feature_map[key] = tf.io.FixedLenFeature([], tf.float32)
    features = tf.io.parse_single_example(record, features=feature_map)
    return features


def parser_bak(record):
    feature_map = {'label': tf.io.FixedLenFeature([], tf.int64)}
    feature_map['deviceid'] = tf.io.FixedLenFeature([], tf.int64)
    feature_map['app_version'] = tf.io.FixedLenFeature([], tf.int64)
    feature_map['newsid_count'] = tf.io.FixedLenFeature([], tf.int64)
    feature_map['pos_deviceid_count_ratio'] = tf.io.FixedLenFeature([], tf.float32)

    features = tf.io.parse_single_example(record, features=feature_map)
    return features


def tfrecord_pipeline(tfrecord_files, batch_size, epochs):
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.repeat(epochs).map(parser_bak).batch(batch_size)
    return dataset


# def reshape_input(features):
#     test = features['label']
#     label = tf.reshape(features['label'], [-1, 1])
#     reshaped = dict()
#     for key, dim in CATEGORY_FEATURES.items():
#         reshaped[key] = tf.reshape(
#             features[key],
#             [-1, dim])
#     for key, dim in NUMERICAL_FEATURES.items():
#         reshaped[key] = tf.reshape(
#             features[key],
#             [-1, dim])
#     return reshaped, label


def load_tfrecord(file_name, cols_count):
    features = {'x': tf.io.FixedLenFeature([cols_count], tf.float32)}
    data = []
    for s_example in tf.data.TFRecordDataset(file_name):
        example = tf.io.parse_single_example(s_example, features=features)
        data.append(tf.expand_dims(example['x'], 0))
    return tf.concat(data, 0)


def write_feature_map(dateframe, sparse_cols, label=None):
    dense_cols = dateframe[dateframe.columns.difference(sparse_cols).difference([label])].columns
    with open('feature_map.csv', 'w') as f:
        for item in sparse_cols:
            # f.writelines(','.join([str(dateframe[item].max()), item, 'sparse\n']))
            f.writelines(','.join(['1', item, 'sparse\n']))
        for item in dense_cols:
            f.write(','.join(['1', item, 'dense\n']))
        for item in [label]:
            f.write(','.join(['1', item, 'label\n']))


if __name__ == "__main__":
    # data = pd.read_csv('./criteo_sample.txt')
    #
    # sparse_features = ['C' + str(i) for i in range(1, 27)]
    # dense_features = ['I' + str(i) for i in range(1, 14)]
    #
    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )
    # target = ['label']
    #
    # # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])
    # data.to_csv('criteo.csv', index=False)
    # train = pd.read_csv('criteo.csv')
    # cate_cols = [
    #     'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17',
    #     'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'
    # ]
    # write_feature_map(train, cate_cols, 'label')

    prebatch = 1
    parallel_parse = 32
    parallel_reads_per_file = 32
    interleave_cycle = 32
    prefetch_buffer = 4
    shuffle_buffer = 1024
    # CATEGORY_FEATURES, NUMERICAL_FEATURES, VOC_SIZE = get_summary('feature_map.csv')
    # to_tfrecord('train_test.csv', 'train_test.tfrecord', prebatch)

    example_dataset = tfrecord_pipeline('train_test.tfrecord', 1, 1)
    from collections import Iterator, Generator, Iterable

    data_iterator = iter(example_dataset)
    features = next(data_iterator)
    # features, labels = reshape_input(features)
    # features = example_dataset.take(10)
    print(features)

    # next_element = data_iterator.take(1)
    # iterator = iter(data_iterator)
    # while(True):
    # print(next_element)
    print(0)
