from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2	
from sklearn.svm import LinearSVC
import sklearn
import nltk
import re
import codecs

# load the model from disk
filename = 'nlp_model.pkl'
text_clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tfidftransformer.pkl','rb'))
ch2 = pickle.load(open('ch2transformer.pkl', 'rb'))
app = Flask(__name__)

def std_text(text_field):
    text_field = text_field.replace(r"http\S+", "")
    text_field = text_field.replace(r"http", "")
    text_field = text_field.replace(r"@\S+", "")
    text_field = text_field.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    text_field = text_field.replace(r"@", "at")
    text_field = text_field.lower()
    return text_field



@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		message = request.form['message']
		message = std_text(message)
		data = [message]
		vect = ch2.transform(vectorizer.transform(data))
		my_prediction = text_clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
