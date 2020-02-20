#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 CloudBrain <hli@>
#
# Distributed under terms of the CloudBrain license.


"""
convert normal data format to tfrecord
"""


import argparse
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=value))


def to_tfrecord(in_file, out_file, prebatch):
    options = tf.python_io.TFRecordOptions(
                tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(out_file, options=options)
    print('Writing data to .tfrecord format...')
    with open(in_file, 'r') as f:
        dense, sparse, label = (list() for i in range(3))
        for num, line in enumerate(f):
            line = line.strip().split('\t')
            is_test_file = (len(line) == 2)
            dense.extend(float(x) for x in line[0].split(','))
            sparse.extend(int(x) for x in line[1].split(','))
            # assign default label -1 for test data
            label.append(-1 if is_test_file else int(line[2]))
            if ((num + 1) % prebatch) == 0:
                write_tf_example(writer, dense, sparse, label)
                dense, sparse, label = (list() for i in range(3))
            if ((num + 1) % (1024 * prebatch)) == 0:
                print('write %d lines' % num)
    writer.close()


def write_tf_example(writer, dense, sparse, label):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'dense': _float_feature(dense),
                'sparse': _int64_feature(sparse),
                'label': _int64_feature(label),
            }))
    writer.write(example.SerializeToString())


def main(args):
    to_tfrecord(args.in_file, args.out_file, args.prebatch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_file', type=str, required=True)
    parser.add_argument('-o', '--out_file', type=str, required=True)
    parser.add_argument('-s', '--prebatch', type=int, default=256)
    main(parser.parse_args())
