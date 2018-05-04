"""
David R Mohler
EE 5450: Final Project
Spring 2018

Keras Based implementation of Transfer learning for the "Tiny-ImagNet" Dataset
"""
from keras import layers, models, optimizers, Model 
from keras.applications import ResNet50, InceptionV3, Xception, VGG16
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator 
import numpy as np
import cv2 #open CV
import matplotlib.pyplot as plt
import os, shutil 
from scipy.misc import imread

from collections import Counter, OrderedDict


def getTrainClasses():
    # First load wnids
    with open(os.path.join(ImageDir, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]
        wnids.sort()

        keys = wnids

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(ImageDir, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    for label in class_names: #truncate classes to a single label 
        del label[1:]

    class_names = [item for sublist in class_names for item in sublist]
    return keys, class_names #return a list of strings containing class names

def getValidationData(wnids):

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    with open(os.path.join(ImageDir, 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
          img_file, wnid = line.split('\t')[:2]
          img_files.append(img_file)
          val_wnids.append(wnid)
        num_val = len(img_files)

        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
    #    X_val = np.zeros((num_val, 3, 64, 64), dtype=np.float32)
    #    for i, img_file in enumerate(img_files):
    #      img_file = os.path.join(ImageDir, 'val', 'images', img_file)
    #      img = imread(img_file)
    #      if img.ndim == 2:
    #        img.shape = (64, 64, 1)
    #      X_val[i] = img.transpose(2,0,1)
    
    #X_val = np.moveaxis(X_val,1,-1)
    
    return y_val


class ResMod():
    @staticmethod
    def build(): 
        base_model = ResNet50(include_top = False, weights='imagenet')
        
        for layer in base_model.layers:
            layer.trainable=False
            

        x = base_model.output
        x = layers.GlobalMaxPooling2D()(x)
        x = layers.Dense(512,activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(200,activation='softmax')(x)
        
        return predictions,base_model

class VGG16Mod():
    @staticmethod
    def build(): 
        base_model = VGG16(include_top = False, weights='imagenet')
        
        for layer in base_model.layers:
            layer.trainable=False
            

        x = base_model.output
        x = layers.GlobalMaxPooling2D()(x)
        x = layers.Dense(512,activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(200,activation='softmax')(x)
        
        return predictions,base_model

class simpleCNN(): #implementation to establish baseline 
    @staticmethod
    def build():
        model = models.Sequential()

        model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape = (64,64,3)))
        model.add(layers.MaxPool2D((2,2)))

        model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
        model.add(layers.MaxPool2D((2,2)))

        model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
        model.add(layers.MaxPool2D((2,2)))

        model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
        model.add(layers.MaxPool2D((2,2)))


        #Fully Connected or Densely Connected Classifier Network
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5)) #suresh finds this useful, usually do after dense but not here
        model.add(layers.Dense(512,activation='relu'))

    
        #Output layer with a single neuron since it is a binary class problem
        model.add(layers.Dense(200,activation='softmax'))
        model.summary()
        return model



if __name__ == "__main__":
    #dictionary storing different pretrained networks
    MODELS = {
        "resnet50" : ResNet50,
        "inception": InceptionV3,
        "xception" : Xception}

    #extract path to the Tiny-ImageNet files
    ImageDir = os.path.abspath(os.pardir+'/tiny-imagenet-200')
    train_dir = os.path.join(ImageDir,'train')
    val_dir = os.path.join(ImageDir,'val')
    test_dir = os.path.join(ImageDir,'test')
    print("Tiny-ImageNet Directory: ",ImageDir) 

    train_datagen = ImageDataGenerator(rescale= 1./255, # convert pixel integer values
                                           rotation_range = 40,
                                           width_shift_range = 0.2,
                                           height_shift_range = 0.2,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True) 

    val_datagen = ImageDataGenerator(rescale=1./255)


    keys,class_names = getTrainClasses() 

    classLUT = dict(zip(keys,class_names)) #dictionary containing key-value pairs of: 
                                           # "folder_name" : "class_label"

    train_generator = train_datagen.flow_from_directory(train_dir, target_size = (64,64),
                                                         batch_size=50,class_mode='categorical') 

    
    y_val = getValidationData(keys)
    y_val = y_val[:100]

    val_keys = []
    for i in range(100):
        test = [k for k, v in train_generator.class_indices.items() if v == y_val[i]]
        val_keys.append(test[0])
    
    unique_val_keys = list(OrderedDict.fromkeys(val_keys))
    print("number of unique keys: ", len(unique_val_keys))
    print("\n",unique_val_keys)

    #print("\nValidation Data Shape: ",x_val.shape)
    #print("\nValidation Data Type: ",x_val.dtype)
    #print("Validation Label Shape: ",y_val.shape)
    y_val = list(OrderedDict.fromkeys(y_val))
    print( y_val)
  
    print()

    class_count = Counter(y_val)
    
    #directories = sorted(os.listdir(val_dir))
    #for i in range(77):
    #    if os.path.isdir(val_dir+'/'+directories[i]):
    #        os.rename(val_dir+'/'+directories[i],val_dir+'/'+unique_val_keys[i])
        

    ##print(class_count)
    
    #print(directories)
    #print(len(directories))
    #input()


    val_generator = val_datagen.flow_from_directory(val_dir,target_size = (64,64),
                                                   batch_size=20,class_mode='categorical')
    #searches all subfolders for images


    for data_batch, labels_batch in train_generator:
            print("\nData Batch Shape: ",data_batch.shape)
            print("Labels Batch Shape: ",labels_batch.shape)
            #print(data_batch[0,0])
            print(labels_batch[0])
            break

    
    print("Example Class Names:\n",class_names[0:6])

    pred, base_model = VGG16Mod.build()
    model = Model(inputs=base_model.input, outputs = pred)
    #model = simpleCNN.build()


    model.compile(loss="categorical_crossentropy",optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=['accuracy'])
    history = model.fit_generator(train_generator,steps_per_epoch = 500,epochs = 25, validation_data = val_generator)

    #FIX CHANNEL ORDERING IN THE VALIDATION DATA!!!

    model.save('baseline.h5')

    loss, accuracy = model.evaluate(x_val,y_val)

    print("\nNetwork Accuracy (Validation): ",accuracy) 

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs,acc,'b', label = 'Training Accuracy')
    plt.plot(epochs,val_acc,'r',label = 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    #plt.imshow()
    plt.show()

    plt.plot(epochs,loss,'b', label = 'Training loss')
    plt.plot(epochs,val_loss,'r',label = 'Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    #plt.imshow()
    plt.show()