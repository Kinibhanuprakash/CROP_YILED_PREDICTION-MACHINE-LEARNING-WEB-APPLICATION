from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('crop.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("forest_fire.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    predict=model.predict(final)
    output='{}'.format(predict)
    if output>str(0.5):
        return render_template('forest_fire.html',pred='Your yield is in HIGH')
    else:
        return render_template('forest_fire.html',pred='Your yield is LOW.')  
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=6440)
