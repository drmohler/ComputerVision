
import os, shutil
import keras
from keras import layers
from keras import models, Model
from keras import optimizers
from keras.applications import ResNet50, VGG16, Xception
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    keras.__version__

    #Data Preprocessing 
    """
    Currently, our data sits on a drive as JPEG files, so the steps for getting it the images into the network:
    1.Read the picture files.
    2.Decode the JPEG content to RBG grids of pixels.
    3.Convert these into floating point tensors.
    4.Rescale the pixel values (between 0 and 255) to the [0, 1] interval .
    """
    #Setting up the directory paths
    BaseDir = 'C:/DRM_NeuralNets/Neural_Networks/Class Work/CNN/DogsCats/DogsCats' #Change this to match your folders.
    train_dir = os.path.join(BaseDir,'Train')
    val_dir = os.path.join(BaseDir,'Val')
    test_dir = os.path.join(BaseDir,'Test')
    
    train_dogs_dir = os.path.join(train_dir,'TrainDogs')
    train_cats_dir = os.path.join(train_dir,'TrainCats')
    
    val_dogs_dir = os.path.join(val_dir,'ValDogs')
    val_cats_dir = os.path.join(val_dir,'ValCats')

    test_dogs_dir = os.path.join(test_dir,'TestDogs')
    test_cats_dir = os.path.join(test_dir,'TestCats')

    #Check directory paths
    print("Total Training Dog Images = ",len(os.listdir(train_dogs_dir)))
    print("Total Validation Dog Images = ",len(os.listdir(val_dogs_dir)))
    print("Total Test Dog Images = ",len(os.listdir(test_dogs_dir)))

    print("Total Training Cat Images = ",len(os.listdir(train_cats_dir)))
    print("Total Validation Cat Images = ",len(os.listdir(val_cats_dir)))
    print("Total Test Cat Images = ",len(os.listdir(test_cats_dir)))

    #Using ImageDataGenerator setup convert jpeg image to data with scaling.
    #IDG can have 8-26 arguments
    train_datagen = ImageDataGenerator(rescale= 1./255, # convert pixel integer values
                                       rotation_range = 40,
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True) 
    test_datagen = ImageDataGenerator(rescale= 1./255)

    #binary mode create a 1D tensor with only 2 values
    train_generator = train_datagen.flow_from_directory(train_dir, target_size = (299,299),batch_size=20,class_mode='binary') #searches all subfolders for images
    validation_generator = test_datagen.flow_from_directory(val_dir, target_size = (299,299),batch_size=20,class_mode='binary')

    #Examining the output of training generator
    for data_batch, labels_batch in train_generator:
        print("Data Batch Shape: ",data_batch.shape)
        print("Labels Batch Shape: ",labels_batch.shape)
        #print(data_batch[0,0])
        print(labels_batch[0])
        break

  
    #Model Architecture

    base_model = Xception(include_top=False, weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(512,activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(1,activation='sigmoid')(x)

    model = Model(inputs=base_model.input,outputs = predictions)

    model.summary()
   
    #Configure the model for running
    model.compile(loss='binary_crossentropy',optimizer = optimizers.RMSprop(lr =0.00001),metrics=['accuracy'])

    #Train the Model: Fit the model to the Train Data using a batch generator
    history = model.fit_generator(train_generator,steps_per_epoch = 100,epochs = 15,
                        validation_data = validation_generator,validation_steps = 50)#2000 images and 20 batches->1000
    

    #Saving the Trained Model
    model.save('DogsCats.h5')
   

    #Plotting the loss and accuracy
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