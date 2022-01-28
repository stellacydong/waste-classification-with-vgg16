# run by typing python3 main.py in a terminal 
import os
import cv2
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from utils import get_base_url, allowed_file, and_syntax


# %%
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import glob
from skimage.io import imread, imshow
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# %matplotlib inline

# %%
# load model 
from keras.models import Model,load_model, Sequential
model = load_model('best_weights.hdf5')
model.load_weights('best_weights.hdf5')


def predict_prob(number):
    return [number[0],1-number[0]]

def classify(img):

    ans = np.array(list(map(predict_prob, model.predict(img))))

    if ans[0][0] > 0.5:
        result = "the image belongs to Recycle waste category"
    else:
        result = "the image belongs to Organic waste category "

    statement = f'\n The probability of the object being Recycle is {ans[0][0]*100:.7f}%.\n The probability of the object being Organic is {ans[0][1]*100:.7f}%.\n Therefore,' + result + '.'
    
    return statement





# setup the webserver
'''
    coding center code
    port may need to be changed if there are multiple flask servers running on same server
    comment out below three lines of code when ready for production deployment
'''
port = 11121
base_url = get_base_url(port)
#app = Flask(__name__, static_url_path=base_url+'static')
#app.secret_key = "super secret key"

'''
    cv scaffold code
    uncomment below line when ready for production deployment
'''

app = Flask(__name__)

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False
    
    

@app.route('/')
#@app.route(base_url)
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
#@app.route(base_url, methods=['POST'])
def home_post():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('results', filename=filename))
    
    if "filesize" in request.cookies:
        if not allowed_image_filesize(request.cookies["filesize"]):
            print("Filesize exceeded maximum limit")
            return redirect(request.url)
    
    
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)
# #         


@app.route('/uploads/<filename>')
#@app.route(base_url + '/uploads/<filename>')
def results(filename): 
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img,axis=0)
    res = classify(img)
    return render_template('results.html', filename=filename, labels = res)
    
    

       

@app.route('/files/<path:filename>')
#@app.route(base_url + '/files/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)



if __name__ == "__main__":
#    '''
#    coding center code
#    '''
#    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
#    website_url = 'coding.ai-camp.org'
#    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')
#
#    # remove debug=True when deploying it
#    app.run(host = '0.0.0.0', port=port, debug=True)
#    import sys; sys.exit(0)

    '''
    cv scaffold code
    '''
    Only for debugging while developing
    app.run(port=80, debug=True)

