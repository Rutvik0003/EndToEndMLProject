from flask import Flask,render_template,request

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData


application = Flask(__name__)

app = application

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/predict",methods = ["GET","POST"])
def predict():
    if request.method == "GET":
        return render_template("Form.html")
    else:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=int(request.form.get("reading_score")),
            writing_score=int(request.form.get("writing_score"))
        )

        data_df = data.get_data_as_dataframe()

        pred_pipeline = PredictionPipeline()
        results = pred_pipeline.predict_data(data_df)

        return render_template('Form.html', results = results)
        

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
