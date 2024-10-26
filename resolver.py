import time
from http import HTTPStatus
import os
import graphene

from create_model import Model
from google_cloud_vision_handling import get_main_object_image, get_image_main_color
from handle_google_drive import authenticate_drive, download_files_recursively


class QueryScoreAIReturn(graphene.ObjectType):
    """
    QueryScoreAIReturn represents the result of a similarity score query

    Attributes:
        message (str): a message describing the result of the query
        status (int): the status code of the query response
        score (float): the score of similarity between the two images
    """
    message = graphene.String(description="A message describing the result.")
    score = graphene.Float(description="The score returned by the query.")
    status = graphene.Int(description="The status code of the query response.")


class QueryMainObjectReturn(graphene.ObjectType):
    """
    QueryMainObjectReturn represents the result of a query cropping an image around its main clothes

    Attributes:
        message (str): a message describing the result of the query
        status (int): the status code of the query response
        main_object (str): base64 image of the main object of the image
        type (str): type of object of the main object
    """
    message = graphene.String(description="A message describing the result.")
    status = graphene.Int(description="The status code of the query response.")
    main_object = graphene.String()
    type = graphene.String()


class QueryMainColorReturn(graphene.ObjectType):
    """
    QueryMainColorReturn represents the result of a query getting the main color of an image

    Attributes:
        message (str): a message describing the result of the query
        status (int): the status code of the query response
        color: ([int]): the rgb values of the main color of the image
    """
    message = graphene.String(description="A message describing the result.")
    status = graphene.Int(description="The status code of the query response.")
    color = graphene.List(graphene.Int)


import graphene
from http import HTTPStatus


class QueryScoreAI(graphene.ObjectType):
    """
    QueryScoreAI provides various AI-based queries related to images

    Fields:
        compute_likeliness (QueryScoreAIReturn): computes the similarity score between two images
        get_main_color (QueryMainColorReturn): gets the main color of the provided image
        crop_main_object (QueryMainObjectReturn): crops and returns the main object in the provided image
    """

    compute_likeliness = graphene.Field(
        QueryScoreAIReturn,
        base64_first_image=graphene.String(required=True),
        base64_second_image=graphene.String(required=True),
        description="Computes the similarity score between two images."
    )

    def resolve_compute_likeliness(root, info, base64_first_image, base64_second_image):
        """
        resolve the compute_likeliness field
        :param base64_first_image: base64 string of the first image
        :param base64_second_image: base64 string of the second image
        :return the result containing a similarity score, a message, and a status code.
        """
        from handle_likeliness import get_similarity_score
        score = get_similarity_score(models, get_main_object_image(base64_first_image)[0],
                                     get_main_object_image(base64_second_image)[0])
        return QueryScoreAIReturn(
            message='Similarity score calculated successfully.',
            score=float(score),
            status=HTTPStatus.OK
        )

    get_main_color = graphene.Field(
        QueryMainColorReturn,
        base64_image=graphene.String(required=True),
        description="Gets the main color of the provided image."
    )

    def resolve_get_main_color(root, info, base64_image):
        """
        resolve the get_main_color field
        :param base64_image: case64 string of the image
        :return the result containing the main color, a message, and a status code
        """
        r, g, b = get_image_main_color(base64_image)
        return QueryMainColorReturn(
            message='Successfully got main color.',
            color=[r, g, b],
            status=HTTPStatus.OK
        )

    crop_main_object = graphene.Field(
        QueryMainObjectReturn,
        base64_image=graphene.String(required=True),
        description="Crops and returns the main object in the provided image."
    )

    def resolve_crop_main_object(root, info, base64_image):
        """
        resolve the crop_main_object field
        :param base64_image: base64 string of the image
        :return the result containing the cropped main object, its type, a message, and a status code
        """
        main_object, type = get_main_object_image(base64_image)
        return QueryMainObjectReturn(
            message='Successfully cropped the main object.',
            main_object=main_object,
            type=type,
            status=HTTPStatus.OK
        )


service = authenticate_drive()
folder_id = '1z_0ZIU9b4VxBl2KUg2LhBFKb8ci8yM3u'
output_directory = './models'
download_files_recursively(service, folder_id, output_directory)
print('Download completed !')
models = [
    Model(
        path=os.path.join('models', 'clothes', f'4_classes_model.h5')
    ),
    Model(
        path=os.path.join('models', 'shape', f'8_classes_model.h5')
    ),
    Model(
        path=os.path.join('models', 'color', f'12_classes_model.h5')
    )
]

schema = graphene.Schema(query=QueryScoreAI)