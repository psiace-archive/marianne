"""The model manager for marianne"""
# marianne/model.py

import os
import re
import string

import click
import joblib
import nltk
import pandas as pd
from flask import current_app, g
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def get_model():
    if "model" not in g or "tfidf" not in g:
        g.model = joblib.load(
            os.path.join(current_app.config["SPAM_DETECT_MODEL"], "randomforest.model")
        )
        g.tfidf = joblib.load(
            os.path.join(current_app.config["SPAM_DETECT_MODEL"], "randomforest.tfidf")
        )

    return [g.model, g.tfidf]


def close_model(e=None):
    g.pop("model", None)
    g.pop("tfidf", None)


def predict_text(text):
    [model, tfidf] = get_model()
    tftext = tfidf.transform([text])
    pdtext = pd.concat(
        [
            pd.DataFrame([len(text)]),
            pd.DataFrame([len(re.findall("\d{5,}", text))]),
            pd.DataFrame(tftext.toarray()),
        ],
        axis=1,
    )
    return model.predict(pdtext)


def clean_text(text):
    ps = nltk.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words("english")
    remove_punct = "".join(
        [word.lower() for word in text if word not in string.punctuation]
    )
    tokens = re.split("\W+", remove_punct)
    noStop = [ps.stem(word) for word in tokens if word not in stopwords]
    return noStop


def init_model():
    data = pd.read_csv(
        os.path.join(current_app.config["DATA_PATH"], "spam_and_ham_text.csv")
    )
    data = data[["label", "text"]]
    data["length"] = data["text"].apply(lambda x: len(x) - x.count(" "))
    data["number"] = pd.DataFrame(
        data["text"].apply(lambda x: len(re.findall("\d{5,}", x)))
    )
    tfidf_Vector = TfidfVectorizer(analyzer=clean_text)
    Xtfidf_Vector = tfidf_Vector.fit_transform(data["text"])
    x_features = pd.concat(
        [data["length"], data["number"], pd.DataFrame(Xtfidf_Vector.toarray())], axis=1
    )
    rf = RandomForestClassifier(n_estimators=50, max_depth=None, n_jobs=-1)
    model = rf.fit(x_features.values, data["label"])
    joblib.dump(
        model, current_app.config["SPAM_DETECT_MODEL"] + "/" + "randomforest.model"
    )
    joblib.dump(
        tfidf_Vector,
        current_app.config["SPAM_DETECT_MODEL"] + "/" + "randomforest.tfidf",
    )


@click.command("init-model")
def init_model_command():
    """Clear the existing model and create new model."""
    nltk.download("stopwords")
    init_model()
    click.echo("Initialized the model.")


def init_app(app):
    app.teardown_appcontext(close_model)
    app.cli.add_command(init_model_command)
