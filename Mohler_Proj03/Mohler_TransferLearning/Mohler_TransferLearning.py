"""
David R Mohler
EE 5450: Final Project
Spring 2018

Keras Based implementation of Transfer learning for the "Tiny-ImagNet" Dataset
"""
from keras import layers, models, optimizers, Model 
from keras.applications import ResNet50, InceptionV3, Xception
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator 
import numpy as np
import cv2 #open CV
import matplotlib.pyplot as plt
import os, shutil 

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

"""MODIFY TO INCLUDE THE USE OF MY OWN TOP LAYERS""" 
class ResMod():
    def build(): 
        base_model = ResNet50(include_top = False, weights='imagenet')
        
        for layer in base_model.layers:
            layer.trainable=False
            

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512,activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(200,activation='softmax')(x)


        #top_model = models.Sequential()
        #top_model.add(layers.Flatten(input_shape=base_model.output_shape[1:]))
        #top_model.add(layers.Dense(512,activation='relu'))
        #top_model.add(layers.Dropout(0.5))
        #top_model.add(layers.Dense(200,activation='softmax'))
        #top_model.load_weights(top)

        #x = base_model.output
        #pred = top_model(x)
        
        return predictions,base_model


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

    train_generator = train_datagen.flow_from_directory(train_dir, target_size = (200,200),
                                                         batch_size=20,class_mode='categorical') 
    val_generator = val_datagen.flow_from_directory(val_dir,target_size = (200,200),
                                                    batch_size=20,class_mode='categorical')
    #searches all subfolders for images


    for data_batch, labels_batch in train_generator:
            print("\nData Batch Shape: ",data_batch.shape)
            print("Labels Batch Shape: ",labels_batch.shape)
            #print(data_batch[0,0])
            print(labels_batch[0])
            break

    print(train_generator.data_format)
    print("Example Class Names:\n",class_names[0:6])

    pred, base_model = ResMod.build()
    model = Model(inputs=base_model.input, outputs = pred)

    model.compile(loss="categorical_crossentropy",optimizer=optimizers.SGD(momentum=0.9),
                  metrics=['accuracy'])
    history = model.fit_generator(train_generator,steps_per_epoch = 5000,epochs = 5)

    model.save('first_attempt.h5')

    loss, accuracy = model.evaluate_generator(val_generator)

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