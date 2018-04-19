# USAGE
# python shallownet_animals.py --dataset ../datasets/animals

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from keras.preprocessing.image import ImageDataGenerator as IDG 
from pyimagesearch.datasets import SimpleDatasetLoader
from networks.nn.MohlerNet import MohlerNet1, MohlerNet2  ,MohlerNet3, MohlerNet4 
from keras.optimizers import SGD
from keras.optimizers import Adagrad, Adam, Adamax
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def SaveResults(filename,results):
    if os.path.isfile(filename):
        with open(filename,'a', newline='') as resultFile:
            writer = csv.writer(resultFile,delimiter=',')
            for line in results:
                writer.writerow(line)
    else:
        with open(filename,'w', newline='') as resultFile:
            writer = csv.writer(resultFile,delimiter=',')
            for line in results:
                writer.writerow(line)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42) #Keep random at 42 for consistent evaluations

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
#opt = SGD(lr=0.01)

dataAugmentation = True 
LR,EPS = 0.01, 0.1
#opt = SGD(lr=LR,momentum=0.9) 
#print("Learning Rate: ",LR)#,"\tEpsilon: ",EPS)
#opt = Adagrad(lr=LR,epsilon=EPS) #LR should be 0.01 and eps 0.1 for this optimizer
#opt = Adam()
opt = Adamax(lr=LR)
print("Network Parameters:\n",opt.get_config())
model = MohlerNet4.build(width=32,height=32,depth=3,classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
numEpochs = 50

batch_size = 16

if dataAugmentation == True:
    train_datagen = IDG(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip=True)

    train_datagen.fit(trainX)
    H = model.fit_generator(train_datagen.flow(trainX,trainY,batch_size=batch_size),
                   steps_per_epoch=(2250//batch_size)
                  ,epochs=numEpochs,
                  validation_data=(testX,testY))
else: 
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
	    batch_size=batch_size, epochs=numEpochs, verbose=1)

 #evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch_size)
results = classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=["cat", "dog", "panda"])
print("\n",results)

#Format data to be written to CSV file 
#------------------------------------------------#
test = results.split()
for i in range(len(test)):
    test[i].replace('/','')
test[19:22] = [''.join(test[19:22])]
test = list(filter(None, test)) # fastest
test.insert(0,'')
test = np.reshape(test,(5,5))
NetDict = {
    1 : "ShallowNet",
    2 : "MohlerNet2",
    3 : "MohlerNet3",
    4 : "MohlerNet4" 
    }

OptDict = {
    1:"SGD",
    2:"AdaGrad",
    3:"Adam" ,
    4:"Adamax"
    }
Network = NetDict[4]
Optimizer = OptDict[4]

filename = Network+"_opt-"+Optimizer+"tests"
fcsv = filename+".csv"
SaveResults(fcsv,test) #write results to file
print("Results saved as: ",filename) 
#END RESULTS STORAGE
#-----------------------------------------------------------------#

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure(filename)
plt.plot(np.arange(0, numEpochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, numEpochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, numEpochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, numEpochs), H.history["val_acc"], label="val_acc")
plt.title(Network+ " Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(filename)
plt.show()

    