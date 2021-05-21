from flask import Flask,request,render_template
import numpy as np
app=Flask(__name__)

@app.route('/')

def home():
	return render_template('index.html')
	
@app.route('/predict', methods=['POST'])

def predict():
	import joblib
	model=joblib.load('ModelSocial.ml')
	int_features=[float(i) for i in request.form.values()]
	final_features=[np.array(int_features)]
	clientachate=[0,1]
	final_features=np.array([final_features]).reshape(1,2)
	prediction=model.predict(final_features)
	if(model.predict(final_features)==1):
		prediction='a cliqué sur la publicité.'
	else :
		prediction='n\'a pas cliqué sur la publicité.'
	return render_template('index.html',prediction_text='L\'utilisateur {}'.format(prediction))
if __name__=="__main__":
	app.run(debug=True)
	
	
	