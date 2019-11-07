#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import helper
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

data_dir = './data'
helper.download_extract(data_dir)
celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
celeba_dataset.shape


# In[2]:


from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D ,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():

    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

 
        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        
        model.add(Dense(1024 * 4 * 4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((4, 4, 1024)))
        
        
        model.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        
      
        model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
      
        model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        
        model.add(Conv2DTranspose(self.channels, kernel_size=4, strides=2, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        
        model.add(Conv2D(128, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.02))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.02))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.02))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(1024, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.02))
        model.add(Dropout(0.25))

        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


    def train(self, epochs, batch_size=256):

        steps = 0
        noise_fix = np.random.normal(0, 1, (9, self.latent_dim))

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for batch_images in celeba_dataset.get_batches(batch_size):


                batch_images*=2


                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                out_noise = np.random.normal(0, 1, (9, self.latent_dim))
                out_imgs = self.generator.predict(out_noise)

                d_loss_real = self.discriminator.train_on_batch(batch_images, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
                if steps % 100 == 0:
                        fixed_img = self.generator.predict(noise_fix)
                        plt.imshow(helper.images_square_grid(out_imgs))
                        plt.show()
                        plt.imshow(helper.images_square_grid(fixed_img))
                        plt.show()

                g_loss = self.combined.train_on_batch(noise, valid)

                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (steps, d_loss[0], 100*d_loss[1], g_loss))
                steps+=1
                print("epoch = " ,epoch)
                
            epoch+=1


# In[ ]:


from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import keras as K
K.backend.clear_session() 


# In[ ]:


if __name__ == '__main__':
    bgan = DCGAN()
    bgan.train(epochs=30, batch_size=128)


# In[5]:


for i in range(500):
    noise = np.random.normal(0, 1, (9, 100))
    plt.imshow(helper.images_square_grid(bgan.generator.predict(noise)))
    plt.axis("off")
    plt.savefig("./data/face_images_2/ %d_image.png" % i)
    plt.show()


# In[6]:


bgan.train(epochs=10, batch_size=128)


# In[ ]:




