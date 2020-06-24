# -*- coding: utf-8 -*-

import glob
import os
import tensorflow as tf
import numpy as np
import h5py
from PIL import Image 
import scipy.misc
import scipy.ndimage 

FLAGS = tf.app.flags.FLAGS

def make_data(sess,data,label):
    if FLAGS.train:
        savepath = os.path.join(os.getcwd(),'{}/train.h5'.format(FLAGS.checkpoint_dir))
    else:
        savepath = os.path.join(os.getcwd(),'{}/test.h5'.format(FLAGS.checkpoint_dir))
    with h5py.File(savepath,'w') as hf:
        hf.create_dataset('data',data = data)
        hf.create_dataset('label',data = label)

def read_data(path):
    with h5py.File(path,'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data,label
  
def prepare_data(sess,dataset):
    if FLAGS.train:
        data_dir = os.path.join(os.getcwd(),dataset)
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = sorted(glob.glob(os.path.join(data_dir, "*.bmp")))
    
    return data

def modcrop(image,scale):
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image
    
def preprocess(path,scale):
    image = Image.open(path).convert('L')
    (width, height) = image.size
    label_ = np.array(list(image.getdata())).astype(np.float).reshape((height, width)) / 255
    label_ = modcrop(label_, scale)
    image.close()

    cropped_image = Image.fromarray(label_)
  
    (width, height) = cropped_image.size
    new_width, new_height = int(width / scale), int(height / scale)
    scaled_image = cropped_image.resize((new_width, new_height), Image.ANTIALIAS)
    cropped_image.close()

    (width, height) = scaled_image.size
    input_ = np.array(list(scaled_image.getdata())).astype(np.float).reshape((height, width))

    return input_, label_

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def array_image_save(array, image_path):
    image = Image.fromarray(array)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(image_path)
    print("Saved image: {}".format(image_path))

def train_input_setup(config):
    sess = config.sess
    image_size, label_size, scale, stride, c_dim= config.image_size, config.label_size, config.scale, config.stride, config.c_dim
    data = prepare_data(sess,dataset=config.data_dir)
    print(image_size)
    print(label_size)
    print(scale)
    print(stride)
    print(c_dim)
    
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(image_size - label_size) // 2
    label_padding = label_size // scale
    
    for i in range(len(data)):
        input_,label_ = preprocess(data[i],scale)
        if len(input_.shape) == 3:
            h,w,_ = input_.shape
        else:
            h,w = input_.shape
        
        for x in range(0,h - image_size - 2 * padding + 1,stride):
            for y in range(0, w - image_size - 2 * padding + 1,stride):
                sub_input = input_[x + padding : x + padding + image_size,y + padding : y + padding + image_size]
                x_loc,y_loc = x + label_padding,y + label_padding
                sub_label = label_[x_loc * scale : x_loc * scale + label_size,y_loc * scale : y_loc * scale + label_size]
                
                sub_input = sub_input.reshape([image_size,image_size,c_dim])
                sub_label = sub_label.reshape([label_size,label_size,c_dim])
                
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
        arrdata = np.asarray(sub_input_sequence)
        arrlabel = np.asarray(sub_label_sequence)
        
        make_data(sess,arrdata,arrlabel)
        
def test_input_setup(config):
    sess = config.sess
    image_size,label_size,scale,stride,c_dim = config.image_size,config.label_size,config.scale,config.stride,config.c_dim
    print(image_size)
    print(label_size)
    print(scale)
    print(stride)
    print(c_dim)
  
    data = prepare_data(sess,dataset='Test')
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(image_size - label_size) // 2
    label_padding = label_size // scale
    
    input_, label_ = preprocess(data[FLAGS.pic_idx],scale)

    if len(input_.shape) == 3:
        h, w, _ = input_.shape
    else:
        h, w = input_.shape

    nx, ny = 0, 0
    for x in range(0, h - image_size - padding + 1, stride):
        nx += 1
        ny = 0
        for y in range(0, w - image_size - padding + 1, stride):
            ny += 1
            sub_input = input_[x + padding : x + padding + image_size,y + padding : y + padding + image_size]
            x_loc,y_loc = x + label_padding,y + label_padding
            sub_label = label_[x_loc * scale : x_loc * scale + label_size,y_loc * scale : y_loc * scale + label_size]
                
            sub_input = sub_input.reshape([image_size,image_size,c_dim])
            sub_label = sub_label.reshape([label_size,label_size,c_dim])
                
            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)

    arrdata = np.asarray(sub_input_sequence)
    arrlabel = np.asarray(sub_label_sequence)

    make_data(sess, arrdata, arrlabel)

    return nx, ny
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    