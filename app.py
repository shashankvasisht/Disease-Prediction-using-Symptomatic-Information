from flask import Flask, request, jsonify
from flask_cors import CORS
import train
import pickle

loaded_model = pickle.load(open('decision_tree_classifier.pickle', 'rb'))

a = train.column_headings

import numpy as np



app = Flask(__name__)
CORS(app, resource = {r"/symptoms/*" : {"origin" : "*"}})

@app.route("/symptoms", methods = ["POST"])
def syms():

	vect = np.zeros(len(a)-1)
	symptoms = [ str(s).lower().replace(" ", "_") for s in request.form.getlist('symptoms[]')]


	# if succeeds
	for ix in symptoms:
		x = a.index(ix)
		vect[x] = 1


	#vect = np.array(vect).reshape(1,len(vect))

	bimari = loaded_model.predict([vect])[0]



	resp = {
		'desease' : bimari 
	}

	# if error
	# resp = {
	# 	'error' : 'Could not do it Mother Fucker !'
	# }
	return jsonify(resp)

def main():
	app.run(port = 8000, debug = True)

if __name__ == "__main__":
	main()