import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from vfv import json_parser
from PIL import Image, ImageDraw, ImageFont
import os
from dotenv import load_dotenv, find_dotenv
from algorithms import ImageSimilarityAlgorithms, ImageColorAlgorithms, ImageDegradationAlgorithms


PROJECT_ROOT = os.path.dirname(find_dotenv())


class WordImage(ImageColorAlgorithms):
    """
    A class for representing a word image.
    """
    def __init__(self, image: np.ndarray=np.array([])):
        """
        Initialize the WordImage with an optional image.

        Args:
            image (np.ndarray, optional): The image to set. Defaults to an empty array.
        """
        super().__init__()
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
    def __init__(self, pdf_path: str, page_number: int, bounding_box: list):
        """
        Initialize the WordImageExtractor with the path to the PDF, page number, polygon, DPI, and debug flag.

        Args:
            pdf_path (str): The path to the PDF document.
            page_number (int): The page number in the PDF document (0-based index).
            bounding_box (list): A list of coordinates representing the bounding box (x, y, width, height).
        """
        super().__init__()
        self.pdf_path = pdf_path
        self.page_number = page_number
        self.bounding_box = bounding_box
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
        binary = self._convert_to_binary(cropped_img)

        self.set_image(binary)

    def remove_specks(self, min_size: int = 30):
        """
        Remove small islands (connected components) of black pixels from the image.
        
        Args:
            min_size: Minimum size (in pixels) for a connected component to be kept
        """
        output = self._remove_specks(self.get_image(), min_size)
        self.set_modified_image(output)


class WordImageRenderer(WordImage):
    """
    A class for rendering a word as an image of the same size as the polygon.
    """
    def __init__(self, word: str, width: int, height: int, typeface: str):
        """
        Initialize the WordImageRenderer with word and polygon details.

        Args:
            word (str): The text to render.
            width (int): The width of the image, in pixels.
            height (int): The height of the image, in pixels.
            typeface (str): The typeface to use for the rendered image.
        """
        super().__init__()
        self.word = word
        self.width = width
        self.height = height
        self.typeface = typeface
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
        
        self.set_image(rendered_img)
      
    def smudge(self, distance_threshold: float = 1.5):
        """
        Apply a distance-based smudging for natural-looking text degradation.
        
        Args:
            distance_threshold: Distance threshold for smudging (in pixels)
                            Higher values create more smudging
        """
        output = self._smudge(self.get_image(), distance_threshold)
        self.set_modified_image(output)

    def fade(self, distance_threshold: float = 1.5):
        """
        Apply a distance-based fade for more natural text degradation.

        Args:
            distance_threshold: Distance threshold for fade (in pixels)
                            Higher values create more fade

        """
        output = self._fade(self.get_image(), distance_threshold)
        self.set_modified_image(output)


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
        plt.imshow(self.get_extracted_word_image(), cmap='gray')
        plt.title("Extracted Word Image")
        plt.axis('on')  # Keep the box
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.show()

        plt.imshow(self.get_rendered_word_image(), cmap='gray')
        plt.title("Rendered Word Image")
        plt.axis('on')  # Keep the box
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.show()


class ParagraphProcessor:
    """
    A class for processing a paragraph to obtain quality scores for each word.
    """
    def __init__(self, paragraph: str):
        """
        Initialize the ParagraphProcessor with a paragraph.
        """
        pass




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
    A class for processing a document to obtain quality scores for each word.
    """
    def __init__(self, pdf_path: str, json_path: str):
        """
        Initialize the DocumentProcessor with a Document.

        Args:
            pdf_path (str): The path to the PDF document.
            json_path (str): The path to the JSON document.
        """
        super().__init__(pdf_path, json_path)


class WordScorer(SimilarityAlgorithms):
    """
    A class for scoring the similarity of two word images based on their projection histograms and Hu Moments.
    """
    def __init__(self, 
                 extracted_image: np.ndarray, 
                 rendered_image: np.ndarray, 
                 edge_map_method: str = 'canny',
                 projection_bin_width: int = 4):
        """
        Initialize the SimilarityAlgorithms with two images.
        
        Args:
            extracted_image: numpy array of the extracted word image.
            rendered_image: numpy array of the rendered word image.
            edge_map_method: method for extracting the edge map ('canny' or 'sobel').
            projection_bin_width: width of the bins for the projection histograms.
        """
        super().__init__()
        self.extracted_image = self._convert_to_binary(extracted_image)
        self.rendered_image = self._convert_to_binary(rendered_image)
        self.edge_map_method = edge_map_method
        self.projection_bin_width = projection_bin_width

        self.extracted_edge_map = self._extract_edge_map(self.extracted_image)
        self.rendered_edge_map = self._extract_edge_map(self.rendered_image)


        ### TODO: Actually run the scoring algorithms.

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
