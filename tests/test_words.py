import cv2
import numpy as np
import pytest
from vfv.words import ImageAlgorithms

@pytest.fixture
def alg():
    return ImageAlgorithms()

def test_convert_to_grayscale(alg):
    # Create a white BGR image (3-channel)
    bgr_image = np.full((10, 10, 3), 255, dtype=np.uint8)
    gray = alg._convert_to_grayscale(bgr_image)
    # Result should be a 2D array
    assert gray.ndim == 2
    assert gray.shape == (10, 10)
    # White should remain white
    assert np.all(gray == 255)

def test_convert_gray_to_rgb(alg):
    # Create a grayscale image with a gradient
    gray = np.arange(100, dtype=np.uint8).reshape((10, 10))
    rgb = alg._convert_gray_to_rgb(gray)
    # Result should have three channels
    assert rgb.ndim == 3
    assert rgb.shape == (10, 10, 3)
    # Each channel should equal the original grayscale image
    for channel in range(3):
        np.testing.assert_array_equal(rgb[:, :, channel], gray)

def test_convert_to_binary(alg):
    # Create an array with values around the threshold (127)
    gray = np.array([[126, 127, 128]], dtype=np.uint8)
    binary = alg._convert_to_binary(gray)
    # For cv2.THRESH_BINARY, pixels >127 become 255; otherwise 0.
    expected = np.array([[0, 0, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(binary, expected)

def test_extract_edge_map_canny(alg):
    # Create an image with a black rectangle on white background
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 30), (70, 70), (0, 0, 0), -1)  # filled black rectangle
    edges = alg._extract_edge_map(img, method='canny')
    # The edge map from Canny should be binary (0 and 255)
    unique_vals = np.unique(edges)
    assert set(unique_vals).issubset({0, 255})
    # Also, passing an invalid method should raise a ValueError.
    with pytest.raises(ValueError):
        alg._extract_edge_map(img, method='invalid')

def test_extract_edge_map_sobel(alg):
    # Create an image with a black rectangle on white background
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 30), (70, 70), (0, 0, 0), -1)
    edges = alg._extract_edge_map(img, method='sobel')
    # Sobel edge map should be of type uint8 and in range 0-255.
    assert edges.dtype == np.uint8
    assert edges.min() >= 0 and edges.max() <= 255

def test_smudge(alg):
    # Create a synthetic grayscale image: white background with a black horizontal line.
    img = np.ones((50, 50), dtype=np.uint8) * 255
    cv2.line(img, (10, 25), (40, 25), 0, 1)  # draw a black line
    smudged = alg._smudge(img)
    # The output should be binary (only 0 and 255)
    unique_vals = np.unique(smudged)
    assert set(unique_vals).issubset({0, 255})
    # The shape should be preserved.
    assert smudged.shape == img.shape

def test_fade(alg):
    # Create a synthetic grayscale image: white background with a black horizontal line.
    img = np.ones((50, 50), dtype=np.uint8) * 255
    cv2.line(img, (10, 25), (40, 25), 0, 1)
    faded = alg._fade(img)
    # The output should be binary.
    unique_vals = np.unique(faded)
    assert set(unique_vals).issubset({0, 255})
    # The shape should be preserved.
    assert faded.shape == img.shape

def test_remove_specks(alg):
    # Create a 100x100 white image.
    img = np.ones((100, 100), dtype=np.uint8) * 255
    # Add a small speck (1 pixel) at (10,10)
    img[10, 10] = 0
    # Add a larger black square (at least 5 pixels in area)
    img[50:55, 50:55] = 0
    # Remove specks using a min_size threshold that should remove the 1-pixel island.
    cleaned = alg._remove_specks(img, min_size=5)
    # The small speck should now be white.
    assert cleaned[10, 10] == 255
    # The larger component should remain black.
    assert cleaned[52, 52] == 0
