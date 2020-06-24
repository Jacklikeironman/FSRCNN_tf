# -*- coding: utf-8 -*-
from model import FSRCNN

import numpy as np
import tensorflow as tf
import pprint
import os

flags = tf.app.flags
flags.DEFINE_boolean('train',True,'True for train,False for test[True]')
flags.DEFINE_integer('epoch',15000,'Number of epochs[10]')
flags.DEFINE_integer('batch_size',1024,'Number of batch_size[128]')
flags.DEFINE_float('learning_rate',1e-3,'Learning rate[1e-3]')
flags.DEFINE_float('momentum', 0.9, "The momentum value for the momentum SGD [0.9]")
flags.DEFINE_integer('c_dim',1,'Number of channel[1]')
flags.DEFINE_integer('scale',3,'Default scale[3]')
flags.DEFINE_integer('stride',4,'Default stride[4]')
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("output_dir", "result", "Name of test output directory [result]")
flags.DEFINE_string("data_dir", 'Train', "Name of data directory to train on [Train]")
flags.DEFINE_integer('pic_idx',2,'index of image for test[4]')

FLAGS =flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    
    with tf.Session() as sess:
        fsrcnn = FSRCNN(sess, config=FLAGS)
        fsrcnn.run()

if __name__ == '__main__':
    tf.app.run()