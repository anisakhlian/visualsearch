import pickle 
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Model

def extract_features(image_dir):
    all_features = []
    
    vgg = VGG16()
    vgg.layers.pop()
    vgg = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    
    i=0
    for filename in os.listdir(image_dir):
        if i%100==0:
            print(i)
        i+=1
        image = load_img(os.path.join(image_dir,filename), target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = image.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        image_array = preprocess_input(image_array)
        features = model.predict(image_array).flatten()
        all_features.append(features)
        name = filename.split('.')[0]+'.pkl'
        pickle.dump(features, open(name, 'wb'))
        
    return np.stack(all_features)
