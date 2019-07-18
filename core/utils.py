import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model


def load_data():
    all_features = []
    image_names = []
    features_dir = 'all_features'
    for file in os.listdir(features_dir):
        features=pickle.load(open(os.path.join(features_dir, file), 'rb'))
        all_features.append(features.flatten())
        image_names.append(file.split('.')[0])
    return np.stack(all_features), image_names

def get_image_features(image_path):
    vgg = VGG16()
    vgg.layers.pop()
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-1].output)
    
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    plt.imshow(image_array/255.)
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
    image_array = preprocess_input(image_array)
    image_features = vgg.predict(image_array)
    return image_features

def plot_similar_images(ids, image_names, image_dir):
    N = (len(ids)+1)//2 
    figure, axes = plt.subplots(N, 2)
    i=0
    for n in range(N):
        for m in range(2):
            if n==N-1 and m==1:
                break
            path = glob.glob(os.path.join(image_dir,image_names[ids[i]])+'*')[0]
            similar = load_img(path, target_size=(224, 224))
            similar_array = img_to_array(similar)
            axes[n,m].imshow(similar_array/255.)
            i+=1 
    if len(ids)%2!=0:
        figure.delaxes(axes[n,m])
    else:
        path = glob.glob(os.path.join(image_dir,image_names[ids[i]])+'*')[0]
        similar = load_img(path, target_size=(224, 224))
        similar_array = img_to_array(similar)
        axes[n,m].imshow(similar_array/255.)

