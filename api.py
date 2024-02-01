from fastapi import FastAPI
import uvicorn

from banknote import BankNote

import pickle

model_dict=pickle.load(open("random_forest_classifier.pkl", "rb"))

classifier=model_dict['model']
app=FastAPI()

@app.get('/')
def home():
    return {"Message":"Welcome to home page"}

@app.post('/predict')
def predict_note(data:BankNote):
    data=data.dict()
    variance=data['variance']
    skewness=data["skewness"]
    curtosis=data["curtosis"]
    entropy=data["entropy"]


    prediction=classifier.predict([[variance, skewness, curtosis, entropy]])

    if prediction[0]>0.5:
        prediction="Fake Note"

    else:
        prediction="Bank Note"

    return {"The note is predicted to be a:":prediction}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)