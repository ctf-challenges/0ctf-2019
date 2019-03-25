#!/usr/bin/env python3
import os
import sys
import time
from flask import Flask, render_template, request, send_from_directory, url_for
import numpy as np
import pandas as pd
import zipfile
import tempfile
from matplotlib import pyplot as plt
# Custom Networks
from networks.lenet import LeNet
from flask_bootstrap import Bootstrap
import tensorflow as tf

app = Flask(__name__)
Bootstrap(app)
application = app
DIR = os.path.abspath(os.path.dirname(__file__))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predictimg(path,lenet):
    image = plt.imread(path) 
    confidence = lenet.predict(image)[0]
    predicted_class = np.argmax(confidence)
    return  predicted_class, class_names[predicted_class],confidence[predicted_class]
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
graph = tf.get_default_graph()
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and request.form["submit"]==u"upload":
        try:
            f = request.files['file']
            with zipfile.ZipFile(f) as myzip, tempfile.TemporaryDirectory() as t_dir, graph.as_default():
                lenet = LeNet()
                for i in range(8):
                    pathori = os.path.join(DIR, "static", "%d.jpg" % i)
                    path = os.path.join(t_dir, "%d.jpg" % i)
                    with myzip.open("%d.jpg" % i, "r") as ii:
                        with open(path, "wb") as oo:
                            oo.write(ii.read())

                    imageori = plt.imread(pathori) 
                    imagenew = plt.imread(path)
                    err = mse(imageori,imagenew)
                    print(err)
                    if err<200 and err!=0:
                        predictid, predictclass, _ = predictimg(path,lenet)
                        predictidori, predictclassori, _ = predictimg(pathori,lenet)
                        print(predictclass, predictclassori)
                        if predictid == predictidori:
                            name = "id:%d your result is " % i + predictclass
                            break
                        else:
                            continue
                    elif err!=0:
                        name ="id:%d error too much modification, %s" % (i, err)
                        break
                    else:
                        name ="id:%d please do something" % i
                        break
                else:
                    name = "flag{FLAG}"
        except Exception as e:
            name = "error %s" % e
    else:
        name = ""

    return render_template('index.html', name=name,time_val=time.time())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=14577, debug=True)

