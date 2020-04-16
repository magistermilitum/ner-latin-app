import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

from joblib import dump, load
import pandas as pd
import treetaggerwrapper

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter


app = Flask(__name__)
model = load('CRF_python_2.joblib') 

classes="B-PERS", "I-PERS"
classes_2="B-LOC", "I-LOC"

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    lemma= sent[i][2]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'lemma' : lemma,
        'lemma[:2]' : lemma[:2]
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        lemma1 = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:lemma': lemma1,
            '-1:lemma[:2]': lemma1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        lemma1 = sent[i+1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:lemma': lemma1,
            '+1:lemma[:2]': lemma1[:2],
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, lemma, label in sent]
def sent2tokens(sent):
    return [token for token, postag, lemma, label in sent]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process',methods=['POST'])
def process():

	if request.method == 'POST':

		choice = request.form['taskoption']
		
		#text=[str(x) for x in request.form.values()]#request_form es un generador, al entrar texto y crear lista seleccionamos el primer valor

		#text=[str(x) for x in request.form['rawtext']]

		text=request.form['rawtext']

		#text=text[0]

		#int_features = [int(x) for x in request.form.values()]
		
		tagger=treetaggerwrapper.TreeTagger(TAGLANG='la')
		tags=tagger.tag_text(text)
		tags=[x.split("\t") for x in tags]
		tags=[[x[0], x[1], x[2].split()[0]] for x in tags]

		c=[x[0][-3::] for x in tags]#terminación de cada palabra

		d=["UPPER" if x[0][0].isupper() else "LOWER" for x in tags]#capital o minuscula


		acta=[[x[0]] + [x[1]] + [x[2]] + [y] + [z] + ["O"] + ["O"] for x, y, z in zip(tags, d, c)]
		
		extra=[[str(row[0]), row[1], row[2], row[6]] for row in acta]

		n=[x[0] for x  in extra]

		X = [sent2features(extra)]

		y_pred = model.predict(X)

		m=list(zip(n, y_pred[0]))#m contiene una lista de tuplas con entidad y tipo.


		if choice == 'person':

			results=[x[0] for x in m if x[1]=="B-PERS" ]

			num_of_results = len(results)



    #m=" ".join([x[0]+"&emsp;"+x[1] for x in m])

    #return render_template('index.html', prediction_text=m)

	return render_template("index.html",results=results,num_of_results = num_of_results)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)