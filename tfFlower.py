#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import keras
import glob
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[5]:


from __future__ import absolute_import, division, print_function, unicode_literals
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib


# In[6]:


#dataset parameters
MODE = 'folder'
DATASET_PATH = '/Users/atulpc/Desktop/New folder/Study/flower_photos'


# In[7]:


#image parameters
N_CLASSES = 5
IMG_SHAPE = [150,150]
CHANNELS = 3
BATCH_SIZE = 100


# In[8]:


def group_images(dataset_path):
    imagepaths, labels = list(), list()
    try :
        classes = sorted(os.walk(dataset_path).next()[1])
    except Exception:
        classes = sorted(os.walk(dataset_path).__next__()[1])
    for c in classes:
        labels.append(c)
        c_dir = os.path.join(dataset_path,c)
        images = glob.glob(c_dir + '/*.jpg')
        #print("{}: {} Images".format(cl, len(images)))
        train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]
            
        for t in train:
            if not os.path.exists(os.path.join(dataset_path, 'train', c)):
                os.makedirs(os.path.join(dataset_path, 'train', c))
            shutil.move(t, os.path.join(dataset_path, 'train', c))

        for v in val:
            if not os.path.exists(os.path.join(dataset_path, 'val', c)):
                os.makedirs(os.path.join(dataset_path, 'val', c))
            shutil.move(v, os.path.join(dataset_path, 'val', c))
    


# In[9]:


# reading the dataset - two modes, file or folder
def read_images(dataset_path, mode, batch_size):
    
    imagepaths, labels = list(), list()
    if mode == 'file':
        with open(dataset_path) as f:
            data = f.read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(d.split(' ')[1])
    elif mode == 'folder':
        label=0
        try :
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:
            classes = sorted(os.walk(dataset_path).__next__()[1])
        for c in classes:
            c_dir = os.path.join(dataset_path,c)
            try:
                walk = os.walk(c_dir).next()
            except:
                walk = os.walk(c_dir).__next__()
            for sample in walk[2]:
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir,sample))
                    #labels.append(label)
            #label = label+1
    else:
        raise Exception("Unknown mode")
        
    
    #converting to tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    #labels = tf.convert_to_tensor(label, dtype=tf.string)
    """
    image, label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)
    
    #reading images in tf
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels = CHANNELS)
    image = tf.image.resize_images(image,IMG_SHAPE)
    X, Y = tf.train.batch([image, label], batch_size=BATCH_SIZE,)
    """
    return imagepaths


# In[10]:


train_dir = os.path.join(DATASET_PATH,'train')
val_dir = os.path.join(DATASET_PATH,'val')
group_images(DATASET_PATH)
X = read_images(train_dir,'folder',150)


# In[11]:


train_daisy_dir = os.path.join(train_dir, 'daisy')  # directory with our training cat pictures
train_dandelion_dir = os.path.join(train_dir, 'dandelion')  # directory with our training dog pictures
train_roses_dir = os.path.join(train_dir, 'roses')  # directory with our training cat pictures
train_sunflowers_dir = os.path.join(train_dir, 'sunflowers')  # directory with our training dog pictures
train_tulips_dir = os.path.join(train_dir, 'tulips')  # directory with our training cat pictures

val_daisy_dir = os.path.join(val_dir, 'daisy')  # directory with our validation cat pictures
val_dandelion_dir = os.path.join(val_dir, 'dandelion')  # directory with our validation dog pictures
val_roses_dir = os.path.join(val_dir, 'roses')  # directory with our validation dog pictures
val_sunflowers_dir = os.path.join(val_dir, 'sunflowers')  # directory with our validation dog pictures
val_tulips_dir = os.path.join(val_dir, 'tulips')  # directory with our validation dog pictures

total_train = len(os.listdir(train_daisy_dir)) +len(os.listdir(train_dandelion_dir)) +len(os.listdir(train_roses_dir)) +len(os.listdir(train_sunflowers_dir)) +len(os.listdir(train_tulips_dir))

total_val = len(os.listdir(val_daisy_dir))+len(os.listdir(val_dandelion_dir))+len(os.listdir(val_roses_dir))+len(os.listdir(val_sunflowers_dir))+len(os.listdir(val_tulips_dir))

print (total_train,'\t', total_val)


# In[12]:


image_gen_train = ImageDataGenerator(rescale=1./255, rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2, horizontal_flip=True,fill_mode='nearest')
train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,directory=train_dir,shuffle=True,target_size=(IMG_SHAPE,IMG_SHAPE), class_mode='categorical')

image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,directory=val_dir,target_size=(IMG_SHAPE,IMG_SHAPE),class_mode='categorical')


# In[13]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation ='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3), activation ='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128,(3,3), activation ='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128,(3,3), activation ='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(5,activation = 'softmax'),
    
])


# In[14]:


model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])


# In[15]:


model.summary()


# In[16]:


epochs=100
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)


# In[ ]:




