"""The model manager for marianne"""
# marianne/model.py

import os

import click
import joblib
import pandas as pd
from flask import current_app, g
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


def get_model():
    if "model" not in g:
        g.model = joblib.load(current_app.config["SPAM_DETECT_MODEL"])

    return g.model


def close_model(e=None):
    model = g.pop("model", None)

    if model is not None:
        model.close()

def predict_text(text):
    model = get_model()
    count_vect = CountVectorizer()
    cv_text = count_vect.fit_transform([text])
    return model.predict(cv_text)

def init_model():
    data = pd.read_csv(os.path.join(current_app.config["DATA_PATH"], "spam_and_ham_text.csv"))
    data = data[['label', 'text']]
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(data['text'])
    model = RandomForestClassifier(n_estimators=100 )
    model.fit(x_train_counts, data['label'])
    joblib.dump(model, current_app.config["SPAM_DETECT_MODEL"])


@click.command("init-model")
def init_model_command():
    """Clear the existing model and create new model."""
    init_model()
    click.echo("Initialized the model.")


def init_app(app):
    app.teardown_appcontext(close_model)
    app.cli.add_command(init_model_command)
