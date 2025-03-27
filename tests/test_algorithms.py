import cv2
import numpy as np
import pytest
import math
from vfv.algorithms import ImageColorAlgorithms, ImageDegradationAlgorithms, ImageSimilarityAlgorithms

@pytest.fixture
def color_alg():
    return ImageColorAlgorithms()

@pytest.fixture
def degradation_alg():
    return ImageDegradationAlgorithms()

@pytest.fixture
def similarity_alg():
    return ImageSimilarityAlgorithms()

### Color Algorithms
def test_convert_to_grayscale(color_alg):
    # Create a white BGR image (3-channel)
    bgr_image = np.full((10, 10, 3), 255, dtype=np.uint8)
    gray = color_alg._convert_to_grayscale(bgr_image)
    # Result should be a 2D array
    assert gray.ndim == 2
    assert gray.shape == (10, 10)
    # White should remain white
    assert np.all(gray == 255)

def test_convert_gray_to_rgb(color_alg):
    # Create a grayscale image with a gradient
    gray = np.arange(100, dtype=np.uint8).reshape((10, 10))
    rgb = color_alg._convert_gray_to_rgb(gray)
    # Result should have three channels
    assert rgb.ndim == 3
    assert rgb.shape == (10, 10, 3)
    # Each channel should equal the original grayscale image
    for channel in range(3):
        np.testing.assert_array_equal(rgb[:, :, channel], gray)

def test_convert_to_binary(color_alg):
    # Create an array with values around the threshold (127)
    gray = np.array([[126, 127, 128]], dtype=np.uint8)
    binary = color_alg._convert_to_binary(gray)
    # For cv2.THRESH_BINARY, pixels >127 become 255; otherwise 0.
    expected = np.array([[0, 0, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(binary, expected)


### Degradation Algorithms
def test_smudge(degradation_alg):
    # Create a synthetic grayscale image: white background with a black horizontal line.
    img = np.ones((50, 50), dtype=np.uint8) * 255
    cv2.line(img, (10, 25), (40, 25), 0, 1)  # draw a black line
    smudged = degradation_alg._smudge(img)
    # The output should be binary (only 0 and 255)
    unique_vals = np.unique(smudged)
    assert set(unique_vals).issubset({0, 255})
    # The shape should be preserved.
    assert smudged.shape == img.shape

def test_fade(degradation_alg):
    # Create a synthetic grayscale image: white background with a black horizontal line.
    img = np.ones((50, 50), dtype=np.uint8) * 255
    cv2.line(img, (10, 25), (40, 25), 0, 1)
    faded = degradation_alg._fade(img)
    # The output should be binary.
    unique_vals = np.unique(faded)
    assert set(unique_vals).issubset({0, 255})
    # The shape should be preserved.
    assert faded.shape == img.shape

def test_remove_specks(degradation_alg):
    # Create a 100x100 white image.
    img = np.ones((100, 100), dtype=np.uint8) * 255
    # Add a small speck (1 pixel) at (10,10)
    img[10, 10] = 0
    # Add a larger black square (at least 5 pixels in area)
    img[50:55, 50:55] = 0
    # Remove specks using a min_size threshold that should remove the 1-pixel island.
    cleaned = degradation_alg._remove_specks(img, min_size=5)
    # The small speck should now be white.
    assert cleaned[10, 10] == 255
    # The larger component should remain black.
    assert cleaned[52, 52] == 0


### Similarity Algorithms
def test_extract_edge_map_canny(similarity_alg):
    # Create an image with a black rectangle on white background
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 30), (70, 70), (0, 0, 0), -1)  # filled black rectangle
    edges = similarity_alg._extract_edge_map(img, method='canny')
    # The edge map from Canny should be binary (0 and 255)
    unique_vals = np.unique(edges)
    assert set(unique_vals).issubset({0, 255})
    # Also, passing an invalid method should raise a ValueError.
    with pytest.raises(ValueError):
        similarity_alg._extract_edge_map(img, method='invalid')

def test_extract_edge_map_sobel(similarity_alg):
    # Create an image with a black rectangle on white background
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 30), (70, 70), (0, 0, 0), -1)
    edges = similarity_alg._extract_edge_map(img, method='sobel')
    # Sobel edge map should be of type uint8 and in range 0-255.
    assert edges.dtype == np.uint8
    assert edges.min() >= 0 and edges.max() <= 255


def test_hausdorff_distance(similarity_alg):
    # Create two 100x100 white images.
    img1 = np.ones((100, 100), dtype=np.uint8) * 255
    img2 = np.ones((100, 100), dtype=np.uint8) * 255
    # Set a single black pixel in each: at (10,10) in img1, and (20,20) in img2.
    img1[10, 10] = 0
    img2[20, 20] = 0
    # Expected Hausdorff distance is approximately sqrt((10-20)^2 + (10-20)^2) = sqrt(200)
    expected_distance = math.sqrt(200)
    hd = similarity_alg._hausdorff_distance(img1, img2)
    assert np.isclose(hd, expected_distance, atol=0.5)
    # If one image has no black pixels, the function should return infinity.
    img3 = np.ones((100, 100), dtype=np.uint8) * 255
    hd_inf = similarity_alg._hausdorff_distance(img1, img3)
    assert np.isinf(hd_inf)

def test_jaccard_similarity(similarity_alg):
    # Create two small binary images manually.
    img1 = np.array([[0, 255],
                     [255, 255]], dtype=np.uint8)
    img2 = np.array([[0, 0],
                     [255, 255]], dtype=np.uint8)
    # For img1: black pixel at (0,0); for img2: black pixels at (0,0) and (0,1)
    # Intersection: 1, union: 2, so similarity should be 0.5.
    sim = similarity_alg._jaccard_similarity(img1, img2)
    assert np.isclose(sim, 0.5)
    # If both images are completely white, union is zero and similarity should be 0.
    img_white = np.ones((2, 2), dtype=np.uint8) * 255
    sim_white = similarity_alg._jaccard_similarity(img_white, img_white)
    assert sim_white == 0.0

def test_compute_projection_histogram(similarity_alg):
    # Create a 10x10 grayscale image with a constant value (e.g., 200).
    img = np.full((10, 10), 200, dtype=np.uint8)
    vertical_hist, horizontal_hist = similarity_alg._compute_projection_histogram(img, bin_width=4)
    # Expected number of bins is ceil(10/4) = 3 for both.
    assert len(vertical_hist) == 3
    assert len(horizontal_hist) == 3
    # If histograms are non-zero, they should be normalized to sum to 1.
    if vertical_hist.sum() != 0:
        np.testing.assert_almost_equal(vertical_hist.sum(), 1.0)
    if horizontal_hist.sum() != 0:
        np.testing.assert_almost_equal(horizontal_hist.sum(), 1.0)

def test_compute_hu_moments(similarity_alg):
    # Create a simple binary image: a white image with a black square.
    img = np.ones((50, 50), dtype=np.uint8) * 255
    cv2.rectangle(img, (15, 15), (35, 35), 0, -1)
    hu_moments = similarity_alg._compute_hu_moments(img)
    # Hu Moments should be a 1D array of length 7.
    assert isinstance(hu_moments, np.ndarray)
    assert hu_moments.shape == (7,)
    # All values should be finite.
    assert np.all(np.isfinite(hu_moments))

def test_compute_wasserstein_distance(similarity_alg):
    # Create two identical histograms.
    hist1 = np.array([0.2, 0.3, 0.5])
    hist2 = np.array([0.2, 0.3, 0.5])
    dist_same = similarity_alg._compute_wasserstein_distance(hist1, hist2)
    assert np.isclose(dist_same, 0.0)
    # Create two different histograms.
    hist3 = np.array([0.1, 0.3, 0.6])
    dist_diff = similarity_alg._compute_wasserstein_distance(hist1, hist3)
    assert dist_diff > 0

def test_compute_cosine_distance(similarity_alg):
    # Test with identical vectors.
    vec1 = np.array([1, 2, 3], dtype=np.float64)
    vec2 = np.array([1, 2, 3], dtype=np.float64)
    cos_dist_same = similarity_alg._compute_cosine_distance(vec1, vec2)
    assert np.isclose(cos_dist_same, 0.0)
    # Test with one zero vector.
    vec_zero = np.array([0, 0, 0], dtype=np.float64)
    cos_dist_zero = similarity_alg._compute_cosine_distance(vec1, vec_zero)
    assert np.isclose(cos_dist_zero, 1.0)
    # Test with opposite vectors.
    vec3 = np.array([-1, -2, -3], dtype=np.float64)
    cos_dist_opposite = similarity_alg._compute_cosine_distance(vec1, vec3)
    # For opposite vectors, cosine similarity should be -1, so cosine distance = 2.
    assert np.isclose(cos_dist_opposite, 2.0)
