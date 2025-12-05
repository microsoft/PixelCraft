import cv2
import os
from src.tools.mask_lines import get_color


def add_axvline(image_path, points, output_path, axis="x", color="red", thickness=2):
    """
    Add vertical or horizontal lines to an image at specified point coordinates.
    
    :param image_path: Path to the input image.
    :param points: List of points where the lines will be drawn.
    :param output_path: Path to save the output image.
    :param axis: Axis along which to draw the lines ('x' for vertical, 'y' for horizontal).
    :param color: Color of the lines (default is 'red').
    :param thickness: Thickness of the lines (default is 2).
    :return: Modified image with lines drawn.
    """
    # Read the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Get color in BGR format
    bgr_color = get_color(color, bgr=True)

    # Draw lines based on axis
    for point in points:
        if axis == "x":
            if point[0] < 0 or point[0] >= w:
                raise ValueError(f"X coordinate {point[0]} is out of bounds for image width {w}.")
            # Draw vertical line
            cv2.line(image, (point[0], 0), (point[0], h), bgr_color, thickness)
        elif axis == "y":
            if point[1] < 0 or point[1] >= h:
                raise ValueError(f"Y coordinate {point[1]} is out of bounds for image height {h}.")
            # Draw horizontal line
            cv2.line(image, (0, point[1]), (w, point[1]), bgr_color, thickness)
        else:
            raise ValueError("Axis must be 'x' or 'y'.")
    # Save the output image
    cv2.imwrite(output_path, image)
    return image


if __name__ == "__main__":
    image_path = "data/CharXiv/images/2.jpg"
    points = [(100, 200)]
    output_path = "out/test/image_with_lines.jpg"
    add_axvline(
        image_path=image_path,
        points=points,
        output_path=output_path,
        axis="y",
        color="red",
        thickness=2,
    )
