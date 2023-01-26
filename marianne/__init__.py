"""Top-level package for marianne"""
# marianne/__init__.py

import os

from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.jinja_env.filters['zip'] = zip 
    app.config.from_mapping(
        SECRET_KEY="dev",
        METADATA_DATABASE=os.path.join(app.instance_path, "db/metadata.sqlite"),
        SPAM_DETECT_MODEL=os.path.join(app.instance_path, "model/spam_detect"),
        DATA_PATH=os.path.join(app.instance_path, "data"),
        TEMPLATES_AUTO_RELOAD=True,
        CRAWLER_LIMIT=10,
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import search

    app.register_blueprint(search.bp)
    app.add_url_rule("/", endpoint="index")

    from . import db, model, features

    db.init_app(app)
    model.init_app(app)
    features.init_app(app)

    return app
