# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from spectrogram_generator import Loop
from keras import backend as K

self = Loop()

K.set_image_dim_ordering('th')

xTrain, yTrain = self.TrainingDataFromDir()


img_rows = 129
img_cols = 178
num_channel = 1
num_epoch = 10

# Define the number of classes
num_classes = 2

img_data_list = []

# for dataset in data_dir_list:
#     img_list = os.listdir(data_path + '/' + dataset)
#     print ('Loaded the images of dataset-' + '{}\n'.format(dataset))
#     for img in img_list:
#         input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
#         input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
#         input_img_resize = cv2.resize(input_img, (128, 128))
#         img_data_list.append(input_img_resize)

# img_data = np.array(img_data_list)
img_data = xTrain
# img_data = img_data.astype('float32')
# img_data /= 255
print (img_data.shape)

if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        img_data = np.expand_dims(img_data, axis=1)
        print (img_data.shape)
    else:
        img_data = np.expand_dims(img_data, axis=4)
        print (img_data.shape)

else:
    if K.image_dim_ordering() == 'th':
        img_data = np.rollaxis(img_data, 3, 1)
        print (img_data.shape)

# %%
# %%
# Assigning Labels

# Define the number of classes
num_classes = 2

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

names = ['cats', 'dogs', 'horses', 'humans']

# convert class labels to on-hot encoding
# Y = np_utils.to_categorical(labels, num_classes)
Y = yTrain

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Training
hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
# hist = model.fit(img_data, Y, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(img_data, Y))
# hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20,verbose=1, validation_split=0.2)

# Training with callbacks

# visualizing losses and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(num_epoch)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

# %%

# Evaluating the model

# score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
