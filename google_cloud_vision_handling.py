from io import BytesIO
from PIL import Image
from google.cloud import vision
from utils import create_image_from_bytes, base64_to_image, image_to_base64
from google.cloud.vision_v1.types import image_annotator


def compute_rectangle_area(points) -> int:
    """
    Function that compute the area of rectangle defined by 4 points
    :param points: 4 points of the rectangle
    :return: a number representing the area or -1 if there are not 4 points
    """
    if not points[0] or not points[1] or not points[2] or not not points[3]:
        return -1
    return (points[0].x * points[1].y - points[1].x * points[0].y) + (
            points[1].x * points[2].y - points[2].x * points[1].y) + (
            points[2].x * points[3].y - points[3].x * points[2].y) + (
            points[3].x * points[0].y - points[0].x * points[3].y)


def clear_objects(objects: list) -> list:
    """
    Function that removes all objects that are not clothes
    :param objects: list of the objects detected by the object localization of google vision
    :return: new list without the no clothes objects
    """
    new_objects = []
    authorised_objects = ['Top', 'Pants', 'Outerwear', 'Dress', 'Hat']

    for elem in objects:
        print(elem.name)
        if elem.name in authorised_objects:
            new_objects.append(elem)
    return new_objects


def crop_image(vision_image: image_annotator, normalized_vertices: [{str: int}]) -> Image:
    """
    Function that crop the image around the object
    :param vision_image: annotation created by google vision on the image
    :param normalized_vertices: normalized vertices of the object
    :return: Image type og the cropped image
    """
    readable_image = Image.open(BytesIO(vision_image.content))
    width, height = readable_image.size
    left = int(normalized_vertices[0].x * width)
    top = int(normalized_vertices[0].y * height)
    right = int(normalized_vertices[2].x * width)
    bottom = int(normalized_vertices[2].y * height)
    return readable_image.crop((left, top, right, bottom))


def crop_image_around_largest_object(image: Image) -> Image:
    """
    Function that gets the objects in the image from google cloud vision, keeps the bigger and crop the image around it
    :param image: image to crop as an Image type
    :return: Image type of the cropped image
    """
    client = vision.ImageAnnotatorClient()
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    content = img_byte_arr.getvalue()

    vision_image = vision.Image(content=content)
    response = client.object_localization(image=vision_image)

    objects = response.localized_object_annotations
    if not objects:
        return vision_image

    objects = clear_objects(objects)
    largest_object = max(objects, key=lambda obj: compute_rectangle_area(obj.bounding_poly.normalized_vertices))

    return crop_image(vision_image, largest_object.bounding_poly.normalized_vertices), largest_object.name


def get_main_object_image(base64_image: str) -> (str, str):
    """
    Function that get the main object of an image
    :param base64_image: base64 string of the image
    :return: base64 string of the cropped image accompanied by the type of the main object
    """
    image_bytes = base64_to_image(base64_image)
    image = create_image_from_bytes(image_bytes)
    cropped_image, type = crop_image_around_largest_object(image)
    return image_to_base64(cropped_image), type


def get_image_main_color(base64_image: str) -> (int, int, int):
    """
    Function that get the main color of an image
    :param base64_image: base64 string of the image
    :return: r, g and b value of the main color of the image
    """
    image_bytes = base64_to_image(base64_image)
    image = create_image_from_bytes(image_bytes)
    client = vision.ImageAnnotatorClient()
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    content = img_byte_arr.getvalue()

    image = vision.Image(content=content)

    response = client.image_properties(image=image)
    props = response.image_properties_annotation
    color = max(props.dominant_colors.colors, key=lambda color: color.pixel_fraction).color
    return color.red, color.green, color.blue
