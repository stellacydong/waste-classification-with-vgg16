# %%
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import glob
from skimage.io import imread, imshow
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
%matplotlib inline

# %%
# load model 
from keras.models import Model,load_model, Sequential
model = load_model('best_weights.hdf5')
model.load_weights('best_weights.hdf5')

# %%
# Test Case
img = load_img('images/O_13968.jpg', target_size=(224,224))
img = img_to_array(img)
img = img / 255
imshow(img)
plt.axis('off')
img = np.expand_dims(img,axis=0)


def predict_prob(number):
    return [number[0],1-number[0]]
ans = np.array(list(map(predict_prob, model.predict(img))))

print("The probability of being Recycle is {:.7f}%".format(ans[0][0]*100))
print("The probability of being Organic is {:.7f}%".format((ans[0][1])*100))

if ans[0][0] > 0.5:
    print("The image belongs to Recycle waste category")
else:
    print("The image belongs to Organic waste category ")

# %%


# %%
