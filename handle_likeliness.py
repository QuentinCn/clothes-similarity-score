import sys

from sklearn.metrics.pairwise import cosine_similarity
from create_model import Model
from utils import base64_to_image, create_image_from_bytes


def get_similarity_score(models: [Model], image_one, image_two) -> float:
    """
    Function that computes the similarities between 2 images
    :param models: list of model of type Model, will be used to compare the 2 images
    :param image_one: first image as base64
    :param image_two: second image as base64
    :return: a float representing the similarities between the 2 images (1 -> similar, 0 -> not the same)
    """
    input_shape = models[0].get_input_shape()
    try:
        image_one = base64_to_image(image_one)
        image_one = create_image_from_bytes(image_one)
        image_one = image_one.resize((input_shape[0], input_shape[1]))

        image_two = base64_to_image(image_two)
        image_two = create_image_from_bytes(image_two)
        image_two = image_two.resize((input_shape[0], input_shape[1]))
    except Exception as error:
        print(f'error: {error}', file=sys.stderr)
        return -1

    similarities = []
    for model in models:
        similarities.append(cosine_similarity(model.predict(image_one, verbose=0), model.predict(image_two, verbose=0))[0][0])
    return sum(similarities) / len(similarities)
