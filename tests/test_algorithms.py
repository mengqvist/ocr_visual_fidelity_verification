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

def test_extract_edge_map_canny(color_alg):
    # Create an image with a black rectangle on white background
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 30), (70, 70), (0, 0, 0), -1)  # filled black rectangle
    # Convert to binary first
    binary_img = color_alg._convert_to_binary(img)
    edges = color_alg._extract_edge_map(binary_img, method='canny')
    # The edge map from Canny should be binary (0 and 255)
    unique_vals = np.unique(edges)
    assert set(unique_vals).issubset({0, 255})
    # Also, passing an invalid method should raise a ValueError.
    with pytest.raises(ValueError):
        color_alg._extract_edge_map(binary_img, method='invalid')

def test_extract_edge_map_sobel(color_alg):
    # Create an image with a black rectangle on white background
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 30), (70, 70), (0, 0, 0), -1)
    # Convert to binary first
    binary_img = color_alg._convert_to_binary(img)
    edges = color_alg._extract_edge_map(binary_img, method='sobel')
    # Sobel edge map should be of type uint8 and in range 0-255.
    assert edges.dtype == np.uint8
    assert edges.min() >= 0 and edges.max() <= 255

def test_calculate_centroid(similarity_alg):
    # Test 1: Single black pixel
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[50, 50] = 0  # black pixel at (50,50)
    centroid = similarity_alg._calculate_centroid(img)
    # Expect centroid at (50, 50)
    assert np.isclose(centroid[0], 50)
    assert np.isclose(centroid[1], 50)

    # Test 2: Multiple black pixels
    img2 = np.full((100, 100), 255, dtype=np.uint8)
    # Set black pixels at (10,10), (20,20), (30,30)
    img2[10, 10] = 0
    img2[20, 20] = 0
    img2[30, 30] = 0
    centroid2 = similarity_alg._calculate_centroid(img2)
    # Expected centroid is (20,20) (mean of 10,20,30)
    assert np.isclose(centroid2[0], 20)
    assert np.isclose(centroid2[1], 20)

    # Test 3: No black pixels (all white)
    img3 = np.full((100, 100), 255, dtype=np.uint8)
    # Should give the middle of the image
    centroid3 = similarity_alg._calculate_centroid(img3)
    assert np.isclose(centroid3[0], 50)
    assert np.isclose(centroid3[1], 50)


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
def test_projection_histogram(similarity_alg):
    # Create a simple grayscale image (all white)
    img = np.full((10, 10), 255, dtype=np.uint8)
    vertical_hist, horizontal_hist = similarity_alg._projection_histogram(img, bin_width=4)
    # Expected number of bins is ceil(10/4) = 3
    assert len(vertical_hist) == 3
    assert len(horizontal_hist) == 3
    # Histograms should be normalized (sum to 1)
    np.testing.assert_almost_equal(vertical_hist.sum(), 1.0, decimal=5)
    np.testing.assert_almost_equal(horizontal_hist.sum(), 1.0, decimal=5)

def test_hu_moments(similarity_alg):
    # Create an image with a black square on white background
    img = np.full((50, 50), 255, dtype=np.uint8)
    cv2.rectangle(img, (15, 15), (35, 35), 0, -1)
    hu = similarity_alg._hu_moments(img)
    assert hu.shape == (7,)
    assert np.all(np.isfinite(hu))

def test_wasserstein_similarity(similarity_alg):
    # Identical histograms should yield similarity of 1
    hist1 = np.array([0.2, 0.3, 0.5])
    hist2 = np.array([0.2, 0.3, 0.5])
    sim = similarity_alg._wasserstein_similarity(hist1, hist2)
    assert np.isclose(sim, 1.0)
    # Different histograms should yield similarity less than 1
    hist3 = np.array([0.1, 0.3, 0.6])
    sim_diff = similarity_alg._wasserstein_similarity(hist1, hist3)
    assert 0 <= sim_diff < 1.0

def test_cosine_similarity(similarity_alg):
    # Test with identical vectors (similarity should be 1)
    vec1 = np.array([1, 2, 3], dtype=np.float64)
    vec2 = np.array([1, 2, 3], dtype=np.float64)
    sim_same = similarity_alg._cosine_similarity(vec1, vec2)
    assert np.isclose(sim_same, 1.0)
    # Test with one zero vector (should return 0)
    vec_zero = np.array([0, 0, 0], dtype=np.float64)
    sim_zero = similarity_alg._cosine_similarity(vec1, vec_zero)
    assert sim_zero == 0.0
    # Test with opposite vectors: raw cosine = -1, mapped similarity = 0
    vec_opp = np.array([-1, -2, -3], dtype=np.float64)
    sim_opp = similarity_alg._cosine_similarity(vec1, vec_opp)
    assert np.isclose(sim_opp, 0.0)

def test_hu_similarity(similarity_alg):
    # Create two nearly identical images with a black rectangle
    img1 = np.full((50, 50), 255, dtype=np.uint8)
    cv2.rectangle(img1, (10, 10), (40, 40), 0, -1)
    img2 = np.full((50, 50), 255, dtype=np.uint8)
    cv2.rectangle(img2, (12, 12), (38, 38), 0, -1)
    sim_val = similarity_alg._hu_similarity(img1, img2)
    # They should be very similar (close to 1)
    assert sim_val > 0.9
    # Compare with an image with no features (all white)
    img_white = np.full((50, 50), 255, dtype=np.uint8)
    sim_diff = similarity_alg._hu_similarity(img1, img_white)
    assert sim_diff < 0.5

def test_projection_histogram_similarity(similarity_alg):
    # Two identical images should yield high similarity
    img1 = np.full((10, 10), 255, dtype=np.uint8)
    img2 = np.full((10, 10), 255, dtype=np.uint8)
    sim_val = similarity_alg._projection_histogram_similarity(img1, img2)
    assert np.isclose(sim_val, 1.0)
    # A modified image should yield lower similarity
    img3 = img1.copy()
    cv2.line(img3, (0, 5), (9, 5), 0, 1)
    sim_diff = similarity_alg._projection_histogram_similarity(img1, img3)
    assert 0 <= sim_diff < 1.0

def test_robust_chamfer_similarity(similarity_alg):
    # Create two images with a black circle on white background
    img1 = np.full((100, 100), 255, dtype=np.uint8)
    img2 = np.full((100, 100), 255, dtype=np.uint8)
    cv2.circle(img1, (50, 50), 20, 0, -1)
    cv2.circle(img2, (50, 50), 20, 0, -1)
    sim_val = similarity_alg._robust_chamfer_similarity(img1, img2, percentile=95)
    assert 0 <= sim_val <= 1.0
    # If one image is blank, similarity should be 0
    img_blank = np.full((100, 100), 255, dtype=np.uint8)
    sim_blank = similarity_alg._robust_chamfer_similarity(img1, img_blank)
    assert sim_blank == 0.0

def test_jaccard_similarity(similarity_alg):
    # Create two small binary images manually.
    img1 = np.array([[0, 255],
                     [255, 255]], dtype=np.uint8)
    img2 = np.array([[0, 0],
                     [255, 255]], dtype=np.uint8)
    sim_val = similarity_alg._jaccard_similarity(img1, img2)
    # Intersection: 1 pixel; Union: 2 pixels => similarity = 0.5
    assert np.isclose(sim_val, 0.5)
    # If both images are completely white, similarity should be 0
    img_white = np.full((2, 2), 255, dtype=np.uint8)
    sim_white = similarity_alg._jaccard_similarity(img_white, img_white)
    assert sim_white == 0.0
