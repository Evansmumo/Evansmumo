#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the necessary libraries
from flask import Flask,request,jsonify
import joblib


# In[3]:


#load the machine learning model
model=joblib.load('diabetes_model.pkl')


# In[6]:


#create a Flask application
app=Flask('diabetes')


# In[7]:


#define the end for for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    # Make a prediction using the model
    prediction = model.predict([data])[0]
    # Return the prediction as JSON
    return jsonify({'prediction': prediction})


# In[8]:


# Run the Flask application
if "diabetes" == '__main__':
    app.run(debug=True)

