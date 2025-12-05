from PIL import Image
import os
import unittest
from tempfile import TemporaryDirectory

def crop_image(image_path: str, bbox: tuple, output_path: str = None) -> Image.Image:
    """
    Crop an image based on given bounding box coordinates
    
    :param image_path: Path to the input image file
    :param bbox: Bounding box coordinates in format (x_min, y_min, x_max, y_max)
    :param output_path: Optional path to save cropped image (if provided)
    :return: Cropped PIL.Image object
    :raises FileNotFoundError: If input image doesn't exist
    :raises ValueError: For invalid bounding box coordinates
    :raises RuntimeError: For image processing errors
    """
    # Validate image file existence
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        # Open image file
        img = Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {str(e)}")

    # Validate bounding box format
    if len(bbox) != 4:
        raise ValueError("Bounding box requires 4 coordinates (x_min, y_min, x_max, y_max)")
    
    x_min, y_min, x_max, y_max = bbox
    
    # Validate coordinate validity
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"Invalid bounding box {bbox} - coordinates must satisfy x_min < x_max and y_min < y_max")

    # Get image dimensions
    img_width, img_height = img.size
    
    # Validate coordinate range
    if any(coord < 0 for coord in [x_min, y_min]):
        raise ValueError(f"Negative coordinates detected in {bbox}")
    
    if x_max > img_width or y_max > img_height:
        raise ValueError(f"Bounding box {bbox} exceeds image dimensions ({img_width}x{img_height})")

    try:
        # Perform cropping operation
        cropped_img = img.crop((x_min, y_min, x_max, y_max))
    except Exception as e:
        raise RuntimeError(f"Image cropping failed: {str(e)}")

    # Save image if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            cropped_img.save(output_path)
            print(f"Saved cropped image to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save cropped image: {str(e)}")

    return cropped_img

class TestCropImage(unittest.TestCase):
    def setUp(self):
        # Create a test image (100x100 white image)
        self.test_img = Image.new('RGB', (100, 100), color='white')
        self.temp_dir = TemporaryDirectory()
        self.test_path = os.path.join(self.temp_dir.name, 'test_image.jpg')
        self.test_img.save(self.test_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_cropping(self):
        """Test normal cropping operation"""
        with TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'cropped.jpg')
            
            # Crop central 50x50 area
            cropped = crop_image(self.test_path, (25, 25, 75, 75), output_path)
            
            # Verify dimensions
            self.assertEqual(cropped.size, (50, 50))
            
            # Verify file creation
            self.assertTrue(os.path.exists(output_path))

    def test_invalid_bbox(self):
        """Test invalid bounding box handling"""
        with self.assertRaises(ValueError):
            crop_image(self.test_path, (50, 50, 25, 25))  # Reversed coordinates
            
        with self.assertRaises(ValueError):
            crop_image(self.test_path, (0, 0, 150, 100))  # Out of bounds

    def test_nonexistent_file(self):
        """Test handling of missing image file"""
        with self.assertRaises(FileNotFoundError):
            crop_image("non_existent.jpg", (0, 0, 10, 10))

if __name__ == '__main__':
    # unittest.main()

    # Quick usage example without tests
    cropped = crop_image("data/data/CharXiv/images/3.jpg", (10, 10, 290, 290), "pixelcraft/src/tools/cropped.jpg")
    print(f"Cropped image size: {cropped.size}")  # Should print (80, 80) for 100x100 input