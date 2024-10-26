from PIL import Image
import base64
from io import BytesIO


def base64_to_image(base64_string: str) -> bytes:
    """
    Function that convert an image as base64 string to bytes
    :param base64_string: image as a base64 string
    :return: bytes representing the image
    """
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_string)
    return image_bytes


def image_to_base64(image: Image) -> str:
    """
    Function that convert an Image type to base64 string
    :param image: image as an Image type
    :return: base64 string of the image
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


def create_image_from_bytes(image_bytes: bytes) -> Image:
    """
    Function tha convert an image as bytes to an Image type
    :param image_bytes: image as bytes
    :return: image as Image type
    """
    image_stream = BytesIO(image_bytes)
    image = Image.open(image_stream)
    return image
