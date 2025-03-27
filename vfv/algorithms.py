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
        
        # Invert the edges
        edges = cv2.bitwise_not(edges)

        return edges


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
    All similarity metrics return a similarity score between 0 and 1,
    where 1 indicates perfect similarity.
    """
    def __init__(self):
        """
        Initialize the ImageSimilarityAlgorithms.
        """
        super().__init__()

    def _projection_histogram(self, image: np.ndarray, bin_width: int = 4) -> tuple[np.ndarray, np.ndarray]:
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

        # Convert to binary
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

    def _hu_moments(self, image: np.ndarray) -> np.ndarray:
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

        # If the image is empty, return an array of zeros
        if np.count_nonzero(image == 0) == 0:
            return np.zeros(7)

        # Calculate moments
        moments = cv2.moments(image)

        # Compute the seven Hu Moments
        hu_moments = cv2.HuMoments(moments).flatten()

        # Apply log transform to handle large value ranges
        epsilon = 1e-12  # Avoid log(0)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + epsilon)

        return hu_moments

    def _wasserstein_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Computes the Wasserstein similarity between two 1D histograms.
        
        Both histograms must be normalized to sum to 1.
        The bin positions are assumed to be the bin indices.
        The similarity is defined as:
            similarity = 1 / (1 + Wasserstein_distance)
        
        Args:
            hist1 (numpy.ndarray): First normalized histogram.
            hist2 (numpy.ndarray): Second normalized histogram.
        
        Returns:
            float: Wasserstein similarity score between 0 and 1.
        """
        bins = np.arange(len(hist1))
        distance = wasserstein_distance(bins, bins, u_weights=hist1, v_weights=hist2)
        similarity = 1.0 / (1.0 + distance)
        return similarity

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes the cosine similarity between two vectors after L2 normalization.
        
        The raw cosine similarity (dot product of normalized vectors) lies in [-1, 1].
        We map it to a [0, 1] range by:
            similarity = (cosine_similarity + 1) / 2
        
        Args:
            vec1 (numpy.ndarray): First vector.
            vec2 (numpy.ndarray): Second vector.
        
        Returns:
            float: Cosine similarity between 0 and 1. Returns 0 if either vector is zero.
        """
        norm1 = norm(vec1)
        norm2 = norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0

        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        
        cosine_similarity = np.dot(vec1_normalized, vec2_normalized)
        similarity = (cosine_similarity + 1) / 2
        return similarity

    def _hu_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Computes the Hu similarity between two images by comparing their Hu Moments.
        The similarity is defined as the cosine similarity between the Hu Moments vectors.

        Args:
            image1 (numpy.ndarray): First image.
            image2 (numpy.ndarray): Second image.
        
        Returns:
            float: Hu similarity between 0 and 1.
        """
        hu1 = self._hu_moments(image1)
        hu2 = self._hu_moments(image2)
        return float(self._cosine_similarity(hu1, hu2))

    def _projection_histogram_similarity(self, image1: np.ndarray, image2: np.ndarray, bin_width: int = 4) -> float:
        """
        Computes the projection histogram similarity between two images.
        The similarity is defined as the average of the wasserstein similarities between the projection histograms.

        Args:
            image1 (numpy.ndarray): First image.
            image2 (numpy.ndarray): Second image.
            bin_width (int): Width (in pixels) of each bin for the vertical and horizontal projections.
        
        Returns:
            float: Projection histogram similarity between 0 and 1.
        """
        hist1_vertical, hist1_horizontal = self._projection_histogram(image1, bin_width=bin_width)
        hist2_vertical, hist2_horizontal = self._projection_histogram(image2, bin_width=bin_width)
        return float((self._wasserstein_similarity(hist1_vertical, hist2_vertical) + self._wasserstein_similarity(hist1_horizontal, hist2_horizontal)) / 2)

    def _robust_chamfer_similarity(self, image1: np.ndarray, image2: np.ndarray, percentile: int = 95, alpha: float = 10) -> float:
        """
        Computes a robust Chamfer similarity by using a trimmed percentile.
        The underlying distance is converted to similarity as:
            similarity = 1 / (1 + (distance / alpha))
        
        Args:
            image1 (numpy.ndarray): First image.
            image2 (numpy.ndarray): Second image.
            percentile (int): Percentile to use for trimming.
            alpha (float): Scaling factor for the distance.
        
        Returns:
            float: Robust Chamfer similarity score between 0 and 1.
        """
        # Convert to binary if needed
        if len(image1.shape) == 3 or len(image2.shape) == 3:
            image1 = self._convert_to_binary(image1)
            image2 = self._convert_to_binary(image2)
        else:
            image1 = image1.copy()
            image2 = image2.copy()
                
        # If image2 has no foreground (i.e. no black pixels), return 0 similarity.
        if np.count_nonzero(image2 == 0) == 0:
            return 0.0

        # Use foreground pixels (value == 0) from image1 for distance sampling.
        y_coords, x_coords = np.where(image1 == 0)
        if len(y_coords) == 0:
            return 0.0

        dt = cv2.distanceTransform(image2, cv2.DIST_L2, 5)
        sampled_distances = dt[y_coords, x_coords]
        distance = np.percentile(sampled_distances, percentile)
        similarity = 1.0 / (1.0 + (distance / alpha))
        return float(similarity)

    def _chamfer_dispersion_similarity(self, image1: np.ndarray, image2: np.ndarray, outlier_percent: float = 5, beta: float = 1.0) -> float:
        """
        Computes a dispersion-based similarity for the chamfer distances between two images.
        This method removes the top `outlier_percent` of distances (to discard extreme outliers),
        then computes a ratio of the mean of the bottom 80% to the mean of the top 20% and 
        uses this ratio as the similarity score.
        
        Args:
            image1 (numpy.ndarray): First image (e.g., extracted word shape).
            image2 (numpy.ndarray): Second image (e.g., rendered counterpart).
            outlier_percent (float): Percentage of the most distant points to discard.
            beta (float): Scaling factor to adjust the sensitivity.
        
        Returns:
            float: Similarity score between 0 and 1.
        """
        # Convert to binary if needed.
        if len(image1.shape) == 3 or len(image2.shape) == 3:
            image1 = self._convert_to_binary(image1)
            image2 = self._convert_to_binary(image2)
        else:
            image1 = image1.copy()
            image2 = image2.copy()

        # If image2 has no foreground, return 0 similarity.
        if np.count_nonzero(image2 == 0) == 0:
            return 0.0

        # Use foreground (black, value==0) pixels from image1 for distance sampling.
        y_coords, x_coords = np.where(image1 == 0)
        if len(y_coords) == 0:
            return 0.0

        # Compute distance transform on image2.
        dt = cv2.distanceTransform(image2, cv2.DIST_L2, 5)
        sampled_distances = dt[y_coords, x_coords]

        # Trim the top outlier_percent of distances.
        threshold = np.percentile(sampled_distances, 100 - outlier_percent)
        trimmed = sampled_distances[sampled_distances <= threshold]
        if trimmed.size == 0:
            return 0.0

        # devide into top 20% and bottom 80%
        top_20 = trimmed[trimmed > np.percentile(trimmed, 80)]
        bottom_80 = trimmed[trimmed <= np.percentile(trimmed, 80)]
        mean_val_80 = bottom_80.mean()
        mean_val_20 = top_20.mean()

        # Compare the percentages
        similarity = mean_val_80 / mean_val_20

        # Map coefficient of variation to a similarity between 0 and 1.
        return float(similarity)


    def _jaccard_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Computes the Jaccard similarity between two binary images.
        
        Args:
            image1 (numpy.ndarray): First binary image.
            image2 (numpy.ndarray): Second binary image.
        
        Returns:
            float: Jaccard similarity score between 0 and 1 (closer to 1 means more similar).
        """
        # Convert to binary 
        binary1 = self._convert_to_binary(image1)
        binary2 = self._convert_to_binary(image2)

        # Compute the intersection and union of the two binary images
        intersection = np.logical_and(binary1 == 0, binary2 == 0).sum()
        union = np.logical_or(binary1 == 0, binary2 == 0).sum()
        
        if union == 0:
            return 0.0  # Avoid division by zero
        
        return float(intersection / union)
    

    def _black_pixel_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Computes the black pixel similarity between two images.

        Args:
            image1 (numpy.ndarray): First image.
            image2 (numpy.ndarray): Second image.

        Returns:
            float: Black pixel similarity score between 0 and 1.
        """
        return float(np.sum(image1 == 0) / np.sum(image2 == 0))
