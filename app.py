from flask import Flask, render_template, request
import pickle

import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import load_model
from keras.models import Model, load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import keras.models

# Flask utils
from keras.preprocessing import sequence

# create Flask application
app = Flask(__name__)

# read object TfidfVectorizer and model from disk
MODEL_PATH ='model.h5'
model = load_model(MODEL_PATH)
 
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# @app.route('/')
@app.route('/first.html') 
def first():
	return render_template('first.html')

@app.route('/')
# @app.route('/login') 
def login():
	return render_template('login.html')    
    

 
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    error = None
    if request.method == 'POST':
        # message
        msg = request.form['message']
        msg = pd.DataFrame(index=[0], data=msg, columns=['data'])

        # transform data
        new_text = sequence.pad_sequences((tokenizer.texts_to_sequences(msg['data'].astype('U'))), maxlen=547)
          
        # model
        result = model.predict(new_text,batch_size=1,verbose=2)
         
        if result >0.5:
            result = 'Fake'
        else:
            result = 'Real'

        return render_template('index.html', prediction_value=result)
    else:
        error = "Invalid message"
        return render_template('index.html', error=error)
@app.route('/chart') 
def chart():
	return render_template('chart.html')

if __name__ == "__main__":
    app.run()
