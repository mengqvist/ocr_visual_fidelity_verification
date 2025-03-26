import fitz  # PyMuPDF
import cv2
from PIL import Image
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from vfv import json_parser
from PIL import Image, ImageDraw, ImageFont
import math
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import directed_hausdorff
import os
from dotenv import load_dotenv, find_dotenv


PROJECT_ROOT = os.path.dirname(find_dotenv())


class WordImage:
    """
    A class for representing a word image.
    """
    def __init__(self, image: np.ndarray=np.array([])):
        """
        Initialize the WordImage with an optional image.

        Args:
            image (np.ndarray, optional): The image to set. Defaults to an empty array.
        """
        self.image = image
        self.modified_image = image.copy()

    def get_height(self) -> int:
        """
        Get the height of the image.

        Returns:
            int: The height of the image.
        """
        return self.image.shape[0]

    def get_width(self) -> int:
        """
        Get the width of the image.

        Returns:
            int: The width of the image.
        """
        return self.image.shape[1]
    
    def get_dimensions(self) -> tuple[int, int]:
        """
        Get the dimensions of the image (height, width).

        Returns:
            tuple[int, int]: A tuple containing the height and width of the image.
        """
        return self.image.shape[0], self.image.shape[1]
    
    def get_modified_dimensions(self) -> tuple[int, int]:
        """
        Get the dimensions of the modified image (height, width).
        """
        return self.modified_image.shape[0], self.modified_image.shape[1]

    def set_image(self, image: np.ndarray):
        """
        Set the image.

        Args:
            image (np.ndarray): The image to set.
        """
        self.image = image
        self.modified_image = image.copy()

    def get_image(self) -> np.ndarray:
        """
        Get the image.

        Returns:
            np.ndarray: The current image.
        """
        return self.image

    def set_modified_image(self, image: np.ndarray):
        """
        Set a modified image.

        Args:
            image (np.ndarray): The image to set.
        """
        self.modified_image = image

    def get_modified_image(self) -> np.ndarray:
        """
        Get a modified image.

        Returns:
            np.ndarray: The modified image.
        """
        return self.modified_image
    

class WordImageExtractor(WordImage):
    """
    A class for extracting a word image from a PDF document.
    """
    def __init__(self, pdf_path: str, page_number: int, bounding_box: list, debug: bool = False):
        """
        Initialize the WordImageExtractor with the path to the PDF, page number, polygon, DPI, and debug flag.

        Args:
            pdf_path (str): The path to the PDF document.
            page_number (int): The page number in the PDF document (0-based index).
            bounding_box (list): A list of coordinates representing the bounding box (x, y, width, height).
            debug (bool, optional): Flag to enable debug mode. Defaults to False.
        """
        super().__init__()
        self.pdf_path = pdf_path
        self.page_number = page_number

        self.bounding_box = bounding_box
        self.debug = debug
        self._extract_word_image()

    def _extract_word_image(self) -> np.ndarray:
        """
        Extract the image corresponding to the given polygon from the specified page of the PDF.

        Returns:
            A numpy array of the image cropped to the bounding box of the polygon.
        """
        # Open the PDF document.
        pdf_doc = fitz.open(self.pdf_path)
        pdf_page = pdf_doc[self.page_number - 1]
        page_info = pdf_page.get_image_info()[0]
        dpi_x, dpi_y = page_info['xres'], page_info['yres']
        page_width = page_info['bbox'][2] / 72
        page_height = page_info['bbox'][3] / 72
        
        # Make an image
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(dpi_x / 10, dpi_y / 10))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_doc.close()

        # Compute scaling factors from OCR coordinates to image pixels
        scale_x = pix.width / page_width if page_width else 1
        scale_y = pix.height / page_height if page_height else 1

        # Scale bounding box coordinates to image space
        x = self.bounding_box[0] * scale_x
        y = self.bounding_box[1] * scale_y
        width = self.bounding_box[2] * scale_x
        height = self.bounding_box[3] * scale_y

        # Crop the image to the bounding box and convert to numpy array
        cropped_img = np.array(img.crop((x, y, x + width, y + height)))

        # Apply binary thresholding to clean up fuzzy edges
        # You can adjust the threshold value (127) as needed
        _, binary = cv2.threshold(cropped_img, 127, 255, cv2.THRESH_BINARY)

        if self.debug:
            plt.imshow(binary)
            plt.axis('off')
            plt.show()

        self.set_image(binary)

    def remove_specks(self, min_size: int = 30) -> np.ndarray:
        """
        Remove small islands (connected components) of black pixels from the image.
        
        Args:
            min_size: Minimum size (in pixels) for a connected component to be kept
        """
        image = self.get_image()

        # Make sure image is grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # Make sure it's binary (0 for text, 255 for background)
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        
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
        
        # Convert back to RGB if input was RGB
        if len(image.shape) == 3:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        
        self.set_modified_image(output)


class WordImageRenderer(WordImage):
    """
    A class for rendering a word as an image of the same size as the polygon.
    """
    def __init__(self, word: str, width: int, height: int, typeface: str, debug: bool = False):
        """
        Initialize the WordImageRenderer with word and polygon details.

        Args:
            word (str): The text to render.
            width (int): The width of the image, in pixels.
            height (int): The height of the image, in pixels.
            typeface (str): The typeface to use for the rendered image.
            debug (bool, optional): If True, show the rendered image for debugging. Defaults to False.
        """
        super().__init__()
        self.word = word
        self.width = width
        self.height = height
        self.typeface = typeface
        self.debug = debug
        self.kernel_size = 5

        # Assemble paths to the fonts.
        self.font_paths = {
            "courier": os.path.join(PROJECT_ROOT, "fonts", "Courier.ttf"),
            "elite": os.path.join(PROJECT_ROOT, "fonts", "Elite.ttf"),
            "letter_gothic": os.path.join(PROJECT_ROOT, "fonts", "LetterGothicNormal.ttf"),
            "olivetti": os.path.join(PROJECT_ROOT, "fonts", "Olivetti.ttf"),
            "olympia": os.path.join(PROJECT_ROOT, "fonts", "Olympia.ttf"),
            "pica": os.path.join(PROJECT_ROOT, "fonts", "Pica.ttf"),
            "prestige_elite": os.path.join(PROJECT_ROOT, "fonts", "PrestigeElite.ttf"),
            "remington": os.path.join(PROJECT_ROOT, "fonts", "Remington.ttf"),
            "underwood": os.path.join(PROJECT_ROOT, "fonts", "Underwood.ttf")
        }
        if self.typeface not in self.font_paths:
            raise ValueError(f"Font {self.typeface} not found.")

        self._render_word_image()

    def _get_available_fonts(self) -> list[str]:
        """
        Get the available fonts.

        Returns:
            list[str]: A list of the available fonts.
        """
        return list(self.font_paths.keys())

    def _load_font(self, size: int):
        """
        Load a TrueType font.

        Args:
            size (int): The size of the font.

        Returns:
            ImageFont: The loaded font.
        """
        try:
            return ImageFont.truetype(self.font_paths[self.typeface], size)
        except IOError:
            raise IOError(f"Font {self.typeface} not found.")

    def _render_word_image(self) -> np.ndarray:
        """
        Render a word as an image with the size of the polygon.
        
        The rendered word is centered within the bounding box defined by the polygon.
        The function dynamically adjusts the font size so that the text fits within the image,
        and it uses a fallback font if the primary one is not available.
        
        Returns:
            A numpy array of the image with the rendered word.
        """
        # Create a blank grayscale image with a white background.
        rendered_img = Image.new("RGB", (self.width, self.height), color='white')
        draw = ImageDraw.Draw(rendered_img)

        # Start with the maximum possible font size (e.g., the image height).
        font_size = self.height + 10
        font = self._load_font(font_size)
        
        # Measure the text size.
        text_width, text_height = draw.textbbox((0, 0), self.word, font=font)[2:]
        
        # Decrease the font size until the text fits inside the bounding box.
        while (text_width > self.width or text_height > self.height) and font_size > 1:
            font_size -= 1
            font = self._load_font(font_size)
            text_width, text_height = draw.textbbox((0, 0), self.word, font=font)[2:]
        
        # Draw the text in black.
        draw.text((self.width // 2, self.height // 2), self.word, fill='black', font=font, anchor='mm')
        
        # convert to numpy array
        rendered_img = np.array(rendered_img)

        if self.debug:
            plt.imshow(rendered_img)
            plt.axis('off')
            plt.show()
        
        self.set_image(rendered_img)
      
    def smudge(self, distance_threshold: float = 1.5) -> np.ndarray:
        """
        Apply a distance-based smudging for natural-looking text degradation.
        
        Args:
            distance_threshold: Distance threshold for smudging (in pixels)
                            Higher values create more smudging
        
        Returns:
            Binary image with smudged text (black on white)
        """
        image = self.get_image()

        # Convert to grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # Threshold to binary (text as black)
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
               
        # Calculate distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Create output binary image (black text on white background)
        smudged = np.ones_like(binary) * 255
        smudged[dist <= distance_threshold] = 0
        
        # Convert back to RGB to match input format
        smudged_rgb = cv2.cvtColor(smudged, cv2.COLOR_GRAY2RGB)
        
        self.set_modified_image(smudged_rgb)

    def fade(self, distance_threshold: float = 1.5) -> np.ndarray:
        """
        Apply a distance-based fade for more natural text degradation.

        Args:
            distance_threshold: Distance threshold for fade (in pixels)
                            Higher values create more fade

        Returns:
            Binary image with faded text
        """
        image = self.get_image()

        # Convert to grayscale  
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()

        # Invert the image
        inverted_image = cv2.bitwise_not(gray_image)

                # Threshold to binary (text as black)
        _, binary = cv2.threshold(inverted_image, 127, 255, cv2.THRESH_BINARY)
               
        # Calculate distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Create output binary image (black text on white background)
        smudged = np.ones_like(binary) * 255
        smudged[dist <= distance_threshold] = 0
        
        # Convert back to RGB to match input format
        smudged_rgb = cv2.cvtColor(smudged, cv2.COLOR_GRAY2RGB)
        self.set_modified_image(smudged_rgb)


class WordPairProcessor:
    """
    A wrapper class for processing extracting a single word from a PDF document as an image,
    and rendering the corresponding OCR-extracted word as an image of the same size.
    """
    def __init__(self, 
                 pdf_path: str, 
                 page_number: int, 
                 bounding_box: list, 
                 word: str, 
                 typeface: str, 
                 denoise: bool = False,
                 smudge: float = 1.5,
                 debug: bool = False):
        """
        Initialize the WordImageProcessor with PDF details and word information.
        
        Args:
            pdf_path: Path to the PDF file.
            page_number: 1-based page number.
            bounding_box: A flat list of floats [x, y, width, height] defining the bounding box.
            word: The text to render.
            typeface: The typeface to use for the rendered image.
            denoise: If True, denoise the extracted image.
            smudge: How much to smudge (positive value) or fade (negative value) the rendered image.
            debug: If True, show the rendered image for debugging.
        """
        self.pdf_path = pdf_path
        self.page_number = page_number
        self.bounding_box = bounding_box
        self.word = word
        self.typeface = typeface
        self.denoise = denoise
        self.smudge = smudge
        self.debug = debug

        self.extract_obj = WordImageExtractor(pdf_path=self.pdf_path, 
                                              page_number=self.page_number, 
                                              bounding_box=self.bounding_box, 
                                              debug=self.debug)
        
        self.render_obj = WordImageRenderer(word=self.word, 
                                            width=self.extract_obj.get_width(), 
                                            height=self.extract_obj.get_height(),
                                            typeface=self.typeface,
                                            debug=self.debug)

        if self.denoise:
            self.extract_obj.remove_specks()

        if self.smudge > 0:
            self.render_obj.smudge(distance_threshold=self.smudge)
        elif self.smudge < 0:
            self.render_obj.fade(distance_threshold=abs(self.smudge))

        self._validate_images()
    
    def _validate_images(self):
        """
        Validates that the dimensions of the extracted and rendered images are the same.

        Raises:
            ValueError: If the dimensions of the extracted and rendered images do not match.
        """
        if self.extract_obj.get_dimensions() != self.render_obj.get_dimensions():
            raise ValueError("Extracted and rendered images have different dimensions.")
        
        if self.extract_obj.get_modified_dimensions() != self.render_obj.get_modified_dimensions():
            raise ValueError("Extracted and rendered images have different dimensions.")

    def get_extracted_word_image(self) -> np.ndarray:
        """
        Retrieves the extracted word image.

        Returns:
            np.ndarray: The extracted word image as a numpy array.
        """
        return self.extract_obj.get_modified_image()

    def get_rendered_word_image(self) -> np.ndarray:
        """
        Retrieves the rendered word image.

        Returns:
            np.ndarray: The rendered word image as a numpy array.
        """
        return self.render_obj.get_modified_image()

    def show_images(self):
        """
        Shows the extracted and rendered word images.
        """
        plt.imshow(self.get_extracted_word_image())
        plt.title("Extracted Word Image")
        plt.axis('on')  # Keep the box
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.show()

        plt.imshow(self.get_rendered_word_image())
        plt.title("Rendered Word Image")
        plt.axis('on')  # Keep the box
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.show()


class WordScorer:
    """
    A class for scoring the similarity of two word images based on their projection histograms and Hu Moments.
    """
    def __init__(self, 
                 extracted_image: np.ndarray, 
                 rendered_image: np.ndarray, 
                 edge_map_method: str = 'canny',
                 projection_bin_width: int = 4):
        """
        Initialize the WordScorer with two word images.
        
        Args:
            extracted_image: numpy array of the extracted word image.
            rendered_image: numpy array of the rendered word image.
            edge_map_method: method for extracting the edge map ('canny' or 'sobel').
            projection_bin_width: width of the bins for the projection histograms.
        """ 
        self.extracted_image = self._convert_to_binary(extracted_image)
        self.rendered_image = self._convert_to_binary(rendered_image)
        self.edge_map_method = edge_map_method

        self.extracted_edge_map = self._extract_edge_map(self.extracted_image)
        self.rendered_edge_map = self._extract_edge_map(self.rendered_image)

        self.projection_bin_width = projection_bin_width

        self.extracted_projection_histogram = self.compute_projection_histogram(self.extracted_image)
        self.rendered_projection_histogram = self.compute_projection_histogram(self.rendered_image)

        self.extracted_hu_moments = self.compute_hu_moments(self.extracted_image)
        self.rendered_hu_moments = self.compute_hu_moments(self.rendered_image)

        self.projection_distance_height = self.compute_wasserstein_distance(self.extracted_projection_histogram[0], self.rendered_projection_histogram[0])
        self.projection_distance_width = self.compute_wasserstein_distance(self.extracted_projection_histogram[1], self.rendered_projection_histogram[1])

        self.hu_distance = self.compute_cosine_distance(self.extracted_hu_moments, self.rendered_hu_moments)

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Converts the image to grayscale.

        Args:
            image (numpy.ndarray): The image to convert.

        Returns:
            numpy.ndarray: The grayscale image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        
        Args:
            image (numpy.ndarray): Input image.
            method (str): Edge detection method ('canny' or 'sobel').
        
        Returns:
            numpy.ndarray: Edge map image.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == 'canny':
            edges = cv2.Canny(image, 100, 200)
        elif method == 'sobel':
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.magnitude(grad_x, grad_y)
            edges = np.uint8(edges / np.max(edges) * 255)  # Normalize to 0-255
        else:
            raise ValueError("Invalid method. Use 'canny' or 'sobel'.")
        
        return edges

    def hausdorff_distance(self, mode: str = 'all') -> float:
        """
        Computes the Hausdorff distance between two binary images.
        
        Args:
            mode (str): Mode for computing the Hausdorff distance ('all', 'edges').
        
        Returns:
            float: Hausdorff distance.
        """
        if mode not in ['all', 'edges']:
            raise ValueError("Invalid mode. Use 'all' or 'edges'.")

        if mode == 'all':
            points1 = np.column_stack(np.where(self.extracted_image == 0))
            points2 = np.column_stack(np.where(self.rendered_image == 0))
        elif mode == 'edges':
            points1 = np.column_stack(np.where(self.extracted_edge_map == 0))
            points2 = np.column_stack(np.where(self.rendered_edge_map == 0))
        
        if len(points1) == 0 or len(points2) == 0:
            return float('inf')  # Avoid empty input causing errors
        
        # Compute directed Hausdorff distances between black pixel coordinates
        hausdorff_1 = directed_hausdorff(points1, points2)[0]
        hausdorff_2 = directed_hausdorff(points2, points1)[0]
        
        return max(hausdorff_1, hausdorff_2)

    def jaccard_similarity(self, mode: str = 'all') -> float:
        """
        Computes the Jaccard similarity between two binary images.
        
        Args:
            mode (str): Mode for computing the Jaccard similarity ('all', 'edges').

        Returns:
            float: Jaccard similarity score (closer to 1 means more similar).
        """
        if mode not in ['all', 'edges']:
            raise ValueError("Invalid mode. Use 'all' or 'edges'.")

        if mode == 'all':
            intersection = np.logical_and(self.extracted_image == 0, self.rendered_image == 0).sum()
            union = np.logical_or(self.extracted_image == 0, self.rendered_image == 0).sum()
        elif mode == 'edges':
            intersection = np.logical_and(self.extracted_edge_map == 0, self.rendered_edge_map == 0).sum()
            union = np.logical_or(self.extracted_edge_map == 0, self.rendered_edge_map == 0).sum()
        
        if union == 0:
            return 0.0  # Avoid division by zero
        
        return intersection / union

    def compute_projection_histogram(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        # Compute the vertical projection (sum over rows for each column)
        vertical_proj = np.sum(image, axis=0)
        width = image.shape[1]
        num_vertical_bins = math.ceil(width / self.projection_bin_width)
        vertical_hist = np.array([
            np.sum(vertical_proj[i * self.projection_bin_width: min((i + 1) * self.projection_bin_width, width)])
            for i in range(num_vertical_bins)
        ])
        
        # Compute the horizontal projection (sum over columns for each row)
        horizontal_proj = np.sum(image, axis=1)
        height = image.shape[0]
        num_horizontal_bins = math.ceil(height / self.projection_bin_width)
        horizontal_hist = np.array([
            np.sum(horizontal_proj[i * self.projection_bin_width: min((i + 1) * self.projection_bin_width, height)])
            for i in range(num_horizontal_bins)
        ])
        
        # Normalize histograms to sum to 1
        if vertical_hist.sum() != 0:
            vertical_hist = vertical_hist / vertical_hist.sum()
        if horizontal_hist.sum() != 0:
            horizontal_hist = horizontal_hist / horizontal_hist.sum()
        
        return vertical_hist, horizontal_hist

    def compute_hu_moments(self, image: np.ndarray) -> np.ndarray:
        """
        Computes Hu Moments for the input image.
        
        The image is first thresholded to obtain a binary image, then
        moments and Hu moments are computed.
        
        Parameters:
            image (ndarray): Grayscale image.
        
        Returns:
            ndarray: 1D array of 7 Hu Moments.
        """
        # Threshold to create a binary image. Adjust threshold value as needed.
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        
        # Calculate moments
        moments = cv2.moments(binary_image)

        # Compute the seven Hu Moments, which are invariant to translation, scale, and rotation.
        hu_moments = cv2.HuMoments(moments).flatten()

        # Apply log transform to handle large value ranges
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

        return hu_moments

    def compute_wasserstein_distance(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
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

    def compute_cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
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

    def get_distance_scores(self) -> tuple[float, float, float]:
        """
        Returns the distance scores for the word.

        Returns:
            tuple: (projection_distance_height, projection_distance_width, hu_distance)
        """
        return self.projection_distance_height, self.projection_distance_width, self.hu_distance

    def get_distance_scores_as_dict(self) -> dict:
        """
        Returns the distance scores for the word as a dictionary.
        """
        return {"projection_distance_height": self.projection_distance_height, 
                "projection_distance_width": self.projection_distance_width, 
                "hu_distance": self.hu_distance}



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



class Document:
    """
    A class for representing a document.
    """
    def __init__(self, pdf_path: str, json_path: str):
        """
        Initialize the Document with a PDF and JSON path.
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        self.pdf_path = pdf_path
        self.json_path = json_path
        self.json_loader = json_parser.JSONParser(json_path)

    def get_pdf_path(self) -> str:
        """
        Get the path to the PDF document.

        Returns:
            str: The path to the PDF document.
        """
        return self.pdf_path
    
    def get_json_path(self) -> str:
        """
        Get the path to the JSON document.

        Returns:
            str: The path to the JSON document.
        """
        return self.json_path


class DocumentProcessor(Document):
    """
    A class for processing a document.
    """
    def __init__(self, pdf_path: str, json_path: str):
        """
        Initialize the DocumentProcessor with a Document.

        Args:
            pdf_path (str): The path to the PDF document.
            json_path (str): The path to the JSON document.
        """
        super().__init__(pdf_path, json_path)







