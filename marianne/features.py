"""The image features manager for marianne"""
# marianne/features.py

import glob

import click
import cv2

from .imagesearch.descriptor import Descriptor
from .imagesearch.searcher import Searcher


def get_input_features(cd, query):
    return cd.color_moments(query)


def search_image(query):
    RESULTS_ARRAY = []
    SCORE_ARRAY = []
    cd = Descriptor((8, 12, 3))
    features = get_input_features(cd, query)
    searcher = Searcher("marianne/static/unsplash-lite/features/features.csv")
    results = searcher.search(features, "chi2")
    # loop over the results, displaying the score and image name
    for (score, resultID) in results:
        RESULTS_ARRAY.append(resultID)
        SCORE_ARRAY.append(score)
    return RESULTS_ARRAY, SCORE_ARRAY


def init_features():
    # initialize the color descriptor
    cd = Descriptor((8, 12, 3))
    # open the output index file for writing
    output = open("marianne/static/unsplash-lite/features/features.csv", "w")
    # use glob to grab the image paths and loop over them
    for imagePath in glob.glob("marianne/static/unsplash-lite/photos/*.jpg"):
        # extract the image ID (i.e. the unique filename) from the image
        # path and load the image itself
        # imageID = imagePath[imagePath.rfind("/") + 1 :]
        imageID = "unsplash-lite/photos/" + imagePath[imagePath.rfind("/") + 1 :]
        image = cv2.imread(imagePath)
        # describe the image with color_moments
        features = cd.color_moments(image)
        # write the features to file
        features = [str(f) for f in features]
        output.write("%s,%s\n" % (imageID, ",".join(features)))
    # close the index file
    output.close()


@click.command("init-features")
def init_features_command():
    """Clear the existing features and create new features."""
    init_features()
    click.echo("Initialized the features.")


def init_app(app):
    app.cli.add_command(init_features_command)
