import cv2
import numpy as np
from numpy.linalg import norm
import math
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import directed_hausdorff


class ImageColorAlgorithms:
    """
    A class for image color algorithms.
    """
    def __init__(self):
        pass

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Converts the image to grayscale.

        Args:
            image (numpy.ndarray): The image to convert.

        Returns:
            numpy.ndarray: The grayscale image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _convert_gray_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Converts a grayscale image to RGB.

        Args:
            image (numpy.ndarray): The grayscale image to convert.

        Returns:
            numpy.ndarray: The RGB image.
        """
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    def _convert_to_binary(self, image: np.ndarray) -> np.ndarray:
        """
        Converts the image to binary.

        Args:
            image (numpy.ndarray): The image to convert.

        Returns:
            numpy.ndarray: The binary image.
        """
        return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]


class ImageDegradationAlgorithms(ImageColorAlgorithms):
    """
    A class for image degradation algorithms.
    """
    def __init__(self):
        super().__init__()
          
    def _smudge(self, image: np.ndarray, distance_threshold: float = 1.5) -> np.ndarray:
        """
        Apply a distance-based smudging for natural-looking text degradation.
        Assumes that the text is black on white.
        
        Args:
            image (numpy.ndarray): The image to smudge.
            distance_threshold: Distance threshold for smudging (in pixels)
                            Higher values create more smudging
        
        Returns:
            Binary image with smudged text (black on white)
        """

        # Convert to grayscale
        if len(image.shape) == 3:
            gray_image = self._convert_to_grayscale(image)
        else:
            gray_image = image.copy()
        
        # Threshold to binary (text as black)
        binary = self._convert_to_binary(gray_image)
               
        # Calculate distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Create output binary image (black text on white background)
        smudged = np.ones_like(binary) * 255
        smudged[dist <= distance_threshold] = 0
        return smudged

    def _fade(self, image: np.ndarray, distance_threshold: float = 1.5) -> np.ndarray:
        """
        Apply a distance-based fade for more natural text degradation.
        Assumes that the text is black on white.

        Args:
            image (numpy.ndarray): The image to fade.
            distance_threshold: Distance threshold for fade (in pixels)
                            Higher values create more fade

        Returns:
            Binary image with faded text
        """
        # Convert to grayscale  
        if len(image.shape) == 3:
            gray_image = self._convert_to_grayscale(image)
        else:
            gray_image = image.copy()

        # Invert the image
        inverted_image = cv2.bitwise_not(gray_image)

        # Apply the fade
        faded = self._smudge(inverted_image, distance_threshold)
        return faded

    def _remove_specks(self, image: np.ndarray, min_size: int = 30) -> np.ndarray:
        """
        Remove small islands (connected components) of black pixels from the image.
        Assumes that the text is black on white.
        
        Args:
            image (numpy.ndarray): The image to remove specks from.
            min_size: Minimum size (in pixels) for a connected component to be kept

        Returns:
            numpy.ndarray: The image in binary format with specks removed.
        """
        # Make sure image is grayscale
        if len(image.shape) == 3:
            gray_image = self._convert_to_grayscale(image)
        else:
            gray_image = image.copy()
        
        # Make sure it's binary (0 for text, 255 for background)
        binary = self._convert_to_binary(gray_image)
        
        # Find connected components (black regions)
        # The 4 indicates 4-connectivity (only consider adjacent pixels, not diagonals)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(binary), 4, cv2.CV_32S
        )
        
        # Create output image (start with all white)
        output = np.ones_like(binary) * 255
        
        # Copy all components that are large enough to the output image
        # Label 0 is the background (white), so we skip it
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                # This component is large enough to keep
                output[labels == i] = 0
        return output


class ImageSimilarityAlgorithms(ImageColorAlgorithms):
    """
    A class for image similarity algorithms.
    """
    def __init__(self):
        """
        Initialize the ImageSimilarityAlgorithms.
        """
        super().__init__()

    def _extract_edge_map(self, image: np.ndarray, method: str = 'canny') -> np.ndarray:
        """
        Extracts an edge map from a binary or grayscale image.
        Assumes that the text is black on white.
        
        Args:
            image (numpy.ndarray): Input image.
            method (str): Edge detection method ('canny' or 'sobel').
        
        Returns:
            numpy.ndarray: Edge map as a binary image.
        """
        # Ensure the image is grayscale
        if len(image.shape) == 3:
            image = self._convert_to_grayscale(image)
        else:
            image = image.copy()

        # Convert to binary
        binary = self._convert_to_binary(image)
        
        # Apply the edge detection
        if method == 'canny':
            edges = cv2.Canny(binary, 100, 200)
        elif method == 'sobel':
            grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.magnitude(grad_x, grad_y)
            edges = np.uint8(edges / np.max(edges) * 255)  # Normalize to 0-255
        else:
            raise ValueError("Invalid method. Use 'canny' or 'sobel'.")
        
        return edges

    def _hausdorff_distance(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Computes the Hausdorff distance between two binary images.
        
        Args:
            image1 (numpy.ndarray): First binary image.
            image2 (numpy.ndarray): Second binary image.
        
        Returns:
            float: Hausdorff distance.
        """
        points1 = np.column_stack(np.where(image1 == 0))
        points2 = np.column_stack(np.where(image2 == 0))
        
        if len(points1) == 0 or len(points2) == 0:
            return float('inf')  # Avoid empty input causing errors
        
        # Compute directed Hausdorff distances between black pixel coordinates
        hausdorff_1 = directed_hausdorff(points1, points2)[0]
        hausdorff_2 = directed_hausdorff(points2, points1)[0]
        
        return max(hausdorff_1, hausdorff_2)

    def _jaccard_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Computes the Jaccard similarity between two binary images.
        
        Args:
            image1 (numpy.ndarray): First binary image.
            image2 (numpy.ndarray): Second binary image.
        
        Returns:
            float: Jaccard similarity score (closer to 1 means more similar).
        """
        # Convert to binary 
        binary1 = self._convert_to_binary(image1)
        binary2 = self._convert_to_binary(image2)

        # Compute the intersection and union of the two binary images
        intersection = np.logical_and(binary1 == 0, binary2 == 0).sum()
        union = np.logical_or(binary1 == 0, binary2 == 0).sum()
        
        if union == 0:
            return 0.0  # Avoid division by zero
        
        return intersection / union

    def _compute_projection_histogram(self, image: np.ndarray, bin_width: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes normalized vertical and horizontal projection histograms by summing pixel intensities
        using a fixed bin width.
        
        The number of bins is computed as:
        num_vertical_bins = ceil(image_width / bin_width)
        num_horizontal_bins = ceil(image_height / bin_width)
        
        Histograms are normalized to sum to 1.
        
        Parameters:
        image (ndarray): Grayscale image.
        bin_width (int): Width (in pixels) of each bin for the vertical and horizontal projections.
        
        Returns:
        tuple: (vertical_hist, horizontal_hist) as 1D numpy arrays.
        """
        if len(image.shape) == 3:
            image = self._convert_to_grayscale(image)
        else:
            image = image.copy()

        # convert to binary
        binary = self._convert_to_binary(image)

        # Compute the vertical projection (sum over rows for each column)
        vertical_proj = np.sum(binary, axis=0)
        width = binary.shape[1]
        num_vertical_bins = math.ceil(width / bin_width)
        vertical_hist = np.array([
            np.sum(vertical_proj[i * bin_width: min((i + 1) * bin_width, width)])
            for i in range(num_vertical_bins)
        ])
        
        # Compute the horizontal projection (sum over columns for each row)
        horizontal_proj = np.sum(binary, axis=1)
        height = binary.shape[0]
        num_horizontal_bins = math.ceil(height / bin_width)
        horizontal_hist = np.array([
            np.sum(horizontal_proj[i * bin_width: min((i + 1) * bin_width, height)])
            for i in range(num_horizontal_bins)
        ])
        
        # Normalize histograms to sum to 1
        if vertical_hist.sum() != 0:
            vertical_hist = vertical_hist / vertical_hist.sum()
        if horizontal_hist.sum() != 0:
            horizontal_hist = horizontal_hist / horizontal_hist.sum()
        
        return vertical_hist, horizontal_hist

    def _compute_hu_moments(self, image: np.ndarray) -> np.ndarray:
        """
        Computes Hu Moments for the input image.
        
        The image is first thresholded to obtain a binary image, then
        moments and Hu moments are computed.
        
        Parameters:
            image (ndarray): Grayscale image.
        
        Returns:
            ndarray: 1D array of 7 log-transformed Hu Moments.
        """
        # Convert to binary if needed
        if len(image.shape) == 3:
            image = self._convert_to_binary(image)
        else:
            image = image.copy()

        # Calculate moments
        moments = cv2.moments(image)

        # Compute the seven Hu Moments
        hu_moments = cv2.HuMoments(moments).flatten()

        # Apply log transform to handle large value ranges
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-12
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + epsilon)

        return hu_moments

    def _compute_wasserstein_distance(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Computes the Wasserstein distance between two 1D histograms.
        
        Both histograms must be normalized to sum to 1.
        The bin positions are assumed to be the bin indices.
        
        Args:
            hist1: 1D numpy array representing the first histogram.
            hist2: 1D numpy array representing the second histogram.
        
        Returns:
            The Wasserstein distance (a float).
        """
        bins = np.arange(len(hist1))
        distance = wasserstein_distance(bins, bins, u_weights=hist1, v_weights=hist2)
        return distance

    def _compute_cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes the cosine distance between two vectors after L2 normalization.
        
        Cosine distance is defined as 1 minus the cosine similarity of the normalized vectors.
        
        Args:
            vec1: First vector.
            vec2: Second vector.
        
        Returns:
            Cosine distance (a float). Returns 1.0 if either vector is zero.
        """
        # L2 normalize each vector
        norm1 = norm(vec1)
        norm2 = norm(vec2)
        
        # If either vector is zero, return maximum distance.
        if norm1 == 0 or norm2 == 0:
            return 1.0

        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        
        # Compute cosine similarity as the dot product of normalized vectors.
        cosine_similarity = np.dot(vec1_normalized, vec2_normalized)
        
        # Cosine distance is 1 - cosine similarity.
        cosine_distance = 1 - cosine_similarity
        return cosine_distance


# def robust_chamfer_distance(img1, img2, percentile=90):
#     """Computes a robust Chamfer distance by using a trimmed percentile."""
#     img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)[1]
#     img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)[1]
    
#     dt = cv2.distanceTransform(255 - img2, cv2.DIST_L2, 5)
    
#     y_coords, x_coords = np.where(img1 > 0)
#     sampled_distances = dt[y_coords, x_coords]
    
#     # Use a percentile-based approach to reduce outlier influence
#     return np.percentile(sampled_distances, percentile)

# distance = robust_chamfer_distance(img1, img2, percentile=90)
# print("Robust Chamfer Distance:", distance)
