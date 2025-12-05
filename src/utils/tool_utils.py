import base64
import io
from PIL import Image
from typing import List

def resize_max_edge(img: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    resize image to have its longest edge equal to max_size while maintaining aspect ratio
    :param img: PIL Image object
    :param max_size: maximum size for the longest edge
    :return: resized PIL Image object
    """
    width, height = img.size
    # calculate new dimensions
    if width >= height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    return img.resize((new_width, new_height), resample=Image.LANCZOS)

def values_to_pixel(
    values: List[str], coff: float, bias: float, axis: str, axis_sec: int
) -> List[int]:
    """Convert a list of values to their corresponding pixel indices."""
    pixel_values = [int(coff * float(value) + bias) for value in values]
    if axis == "x":
        return [(pixel_value, axis_sec) for pixel_value in pixel_values]
    else:
        return [(axis_sec, pixel_value) for pixel_value in pixel_values]

def encode_image(image):
    if isinstance(image, str):
        # read image from file path
        with open(image, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_image}"
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"
    elif isinstance(image, bytes):
        # read image from bytes
        base64_image = base64.b64encode(image).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"
    else:
        raise TypeError("Unsupported image type. Supported types are: str, PIL Image, bytes.")

def decode_image(base64_string):
    if isinstance(base64_string, dict):
        base64_string = base64_string.get("url", "")
    if base64_string.startswith("data:image/jpeg;base64,"):
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image
