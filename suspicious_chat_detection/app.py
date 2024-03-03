import pandas as pd
from flask import Flask, render_template, request
import suspicious_chat_v2

app = Flask(__name__)

dataset = pd.read_csv("labeled_data.csv")
suspicious = suspicious_chat_v2.SuspiciousDetection(dataset)
messages = []
outputs = []


@app.route('/', methods = ['GET'])
def hello_world():  # put application's code here
    suspicious.load_model()
    return render_template('index.html', name="ML")


@app.route('/', methods = ['POST'])
def predict():
    sentence = request.form['message']
    messages.append(sentence)
    output = suspicious.predict_sentence(sentence)
    outputs.append(output)
    return render_template('index.html', name="ML", prediction=outputs, messages=messages)


if __name__ == '__main__':
    app.run()
