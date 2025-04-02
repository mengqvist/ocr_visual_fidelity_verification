import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from vfv import json_parser
from PIL import Image, ImageDraw, ImageFont
import os
from dotenv import load_dotenv, find_dotenv
from vfv.algorithms import ImageSimilarityAlgorithms, ImageColorAlgorithms, ImageDegradationAlgorithms
from config import TYPEFACE_PATHS   


print(TYPEFACE_PATHS)

PROJECT_ROOT = os.path.dirname(find_dotenv())


class WordImage(ImageDegradationAlgorithms):
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
        # convert to binary, if not already
        if self.image.ndim == 3:
            self.image = self._convert_to_binary(self.image)
        print(f"Image shape: {self.image.shape}")
        return self.image

    def set_modified_image(self, image: np.ndarray):
        """
        Set a modified image.

        Args:
            image (np.ndarray): The image to set.
        """
        # convert to binary, if not already
        if image.ndim == 3:
            image = self._convert_to_binary(image)
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
    def __init__(self, word: str, width: int, height: int, font_size: int, typeface: str, smudge: float=0, centroid: tuple[float, float]=None):
        """
        Initialize the WordImageRenderer with word and polygon details.

        Args:
            word (str): The text to render.
            width (int): The width of the image, in pixels.
            height (int): The height of the image, in pixels.
            font_size (int): The font size to use for the rendered image.
            typeface (str): The typeface to use for the rendered image.
            smudge (float): The amount to smudge the image, in pixels.
            centroid (tuple[float, float]): The centroid of the image, in pixels (x,y). Optional, defaults to None.
        """
        super().__init__()
        self.word = word

        # dimensions and text position
        self.width = width
        self.height = height
        self.centroid = centroid

        # text parameters
        self.font_size = font_size
        self.typeface = typeface   
        self.smudge = smudge

        if not isinstance(self.smudge, (int, float)):
            raise ValueError("Smudge must be a number.")

        if self.centroid is not None:
            # Ensure centroids are integers and within image bounds
            x_centroid, y_centroid = self.centroid
            assert isinstance(x_centroid, (int, float)), "x_centroid must be numeric"
            assert isinstance(y_centroid, (int, float)), "y_centroid must be numeric"
            assert 0 <= x_centroid <= self.width, f"x_centroid {x_centroid} must be between 0 and {self.width}"
            assert 0 <= y_centroid <= self.height, f"y_centroid {y_centroid} must be between 0 and {self.height}"

        if self.typeface not in TYPEFACE_PATHS:
            raise ValueError(f"Font {self.typeface} not found.")

        self._render_word_image()

    def _validate_text_params(self, text_params: dict):
        """
        Validate the text parameters.
        """
        if text_params is None:
            return True
        
        if not isinstance(text_params, dict):
            raise ValueError("Text parameters must be a dictionary.")
        
        if "fontsize" not in text_params:
            raise ValueError("Font size is required.")
        
        if "typeface" not in text_params:
            raise ValueError("Typeface is required.")
        
        if "centroid" not in text_params:
            raise ValueError("Centroid is required.")
        
        if "smudge" not in text_params:
            raise ValueError("Smudge is required.")


        if not isinstance(text_params["fontsize"], (int, float)):
            raise ValueError("Font size must be a number.")
        
        if not isinstance(text_params["typeface"], str):
            raise ValueError("Typeface must be a string.")
        
        if not isinstance(text_params["centroid"], tuple):
            raise ValueError("Centroid must be a tuple.")
        
        if not isinstance(text_params["smudge"], (int, float)):
            raise ValueError("Smudge must be a number.")
        
        if text_params["fontsize"] <= 0:
            raise ValueError("Font size must be greater than 0.")
        
        # if text_params["typeface"] not in []:
        #     raise 

        if text_params["centroid"][0] < 0 or text_params["centroid"][0] > self.width:
            raise ValueError("Centroid x must be between 0 and width.")
        
        if text_params["centroid"][1] < 0 or text_params["centroid"][1] > self.height:
            raise ValueError("Centroid y must be between 0 and height.")    

        if text_params["smudge"] < -5 or text_params["smudge"] > 5:
            raise ValueError("Smudge must be between -5 and 10.")
        
        return True          

    def _get_available_typefaces(self) -> list[str]:
        """
        Get the available typefaces.

        Returns:
            list[str]: A list of the available fonts.
        """
        return list(TYPEFACE_PATHS.keys())

    def _load_font(self, size: int):
        """
        Load a TrueType font.

        Args:
            size (int): The size of the font.

        Returns:
            ImageFont: The loaded font.
        """
        try:
            return ImageFont.truetype(TYPEFACE_PATHS[self.typeface], size)
        except IOError:
            raise IOError(f"Font {self.typeface} not found.")

    # def _render_word_image(self) -> np.ndarray:
    #     """
    #     Render a word as an image with the size of the polygon.
        
    #     The rendered word is centered within the bounding box defined by the polygon.
    #     The function dynamically adjusts the font size so that the text fits within the image,
    #     and it uses a fallback font if the primary one is not available.
        
    #     Returns:
    #         A numpy array of the image with the rendered word.
    #     """
    #     # Create a blank grayscale image with a white background.
    #     rendered_img = Image.new("RGB", (self.width, self.height), color='white')
    #     draw = ImageDraw.Draw(rendered_img)

    #     # Start with the maximum possible font size (e.g., the image height).
    #     font_size = self.height + 10
    #     font = self._load_font(font_size)
        
    #     # Measure the text size.
    #     text_width, text_height = draw.textbbox((0, 0), self.word, font=font)[2:]
        
    #     # Decrease the font size until the text fits inside the bounding box.
    #     while (text_width > self.width or text_height > self.height) and font_size > 1:
    #         font_size -= 1
    #         font = self._load_font(font_size)
    #         text_width, text_height = draw.textbbox((0, 0), self.word, font=font)[2:]
        
    #     # Draw the text in black.
    #     draw.text((self.width // 2, self.height // 2), self.word, fill='black', font=font, anchor='mm')
        
    #     # convert to numpy array
    #     rendered_img = np.array(rendered_img)
        
    #     self.set_image(rendered_img)
        
    def _get_text_width_height(self, rendered_np: np.ndarray) -> tuple[int, int]:
        """
        Get the width and height of the black pixels (text) in the rendered image.

        Returns:
            tuple[int, int]: The width and height of the bounding box that contains all black pixels.
        """
        # convert to to binary
        rendered_np = self._convert_to_binary(rendered_np)

        # Find the coordinates where pixels are black (0)
        y_indices, x_indices = np.where(rendered_np == 0)
        
        # If there are no black pixels, return 0,0
        if x_indices.size == 0 or y_indices.size == 0:
            return 0, 0
        
        # Compute bounding box coordinates
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Width and height of the bounding box (add 1 if you want to count inclusive pixels)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        return width, height


    def _write_text(self, 
                    canvas_width: int,
                    canvas_height: int,
                    font: ImageFont.FreeTypeFont, 
                    text: str, 
                    anchor: str='mm'):
        """
        Write text to an image. Add smudge or fade if requested.
        """
        # create a blank image
        canvas_center = (canvas_width // 2, canvas_height // 2)
        rendered_img = Image.new("RGB", (canvas_width, canvas_height), color='white')
        draw = ImageDraw.Draw(rendered_img)

        # write the text
        draw.text(canvas_center, text, fill='black', font=font, anchor=anchor)
        
        # Convert the rendered image to a NumPy array.
        rendered_np = np.array(rendered_img)

        # apply smudge or fade if requested
        if self.smudge > 0:
            rendered_np = self._smudge(rendered_np, self.smudge)
        elif self.smudge < 0:
            rendered_np = self._fade(rendered_np, self.smudge)

        return rendered_np

    def _render_word_image(self) -> np.ndarray:
        """
        Render a word in an image and align its center of mass with the target centroid.
        
        1. Create a canvas of size (2*self.width, 2*self.height) with a white background.
        2. Render the word (with dynamically adjusted font size) centered on the canvas.
        3. Compute the center of mass of the rendered text using _calculate_centroid.
        4. Crop a region of size (self.width, self.height) such that the rendered centroid is 
        placed at self.centroid (target x, y).
        
        Returns:
            Cropped image as a numpy array.
        """
        # Step 1: Specify a large canvas.
        canvas_width, canvas_height = 2 * self.width, 2 * self.height
                
        # Step 2: Render a first version of the text and get its width and height.
        font_size = self.height + 2
        font = self._load_font(font_size)
        rendered_np = self._write_text(canvas_width, canvas_height, font, self.word)
        text_width, text_height = self._get_text_width_height(rendered_np)

        # Decrease font size until text fits within self.width x self.height.
        while (text_width > self.width or text_height > self.height) and font_size > 1:
            font_size -= 1
            font = self._load_font(font_size)
            rendered_np = self._write_text(canvas_width, canvas_height, font, self.word)
            text_width, text_height = self._get_text_width_height(rendered_np)
        
        # Step 3: Calculate the center of mass from the rendered image.
        # Convert to grayscale for the centroid calculation.
        rendered_gray = self._convert_to_grayscale(rendered_np)  # convert to grayscale
        rendered_centroid = self._calculate_centroid(rendered_gray)  # returns (x, y)
              
        # Step 4: Compute crop coordinates so that the rendered centroid aligns with self.centroid.
        if self.centroid is None:
            target_x, target_y = canvas_width // 2, canvas_height // 2
        else:
            target_x, target_y = self.centroid
        rx, ry = rendered_centroid         # computed center of mass in the large image

        # Determine top-left corner of crop: we want rx to end up at target_x, and ry at target_y.
        crop_x = int(rx - target_x)
        crop_y = int(ry - target_y)
        
        # Ensure the crop stays within the canvas boundaries.
        crop_x = max(0, min(crop_x, canvas_width - self.width))
        crop_y = max(0, min(crop_y, canvas_height - self.height))
        
        cropped_img = rendered_np[crop_y:crop_y + self.height, crop_x:crop_x + self.width]
        
        self.set_image(cropped_img)

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


class WordPairProcessor(ImageSimilarityAlgorithms):
    """
    A wrapper class for processing extracting a single word from a PDF document as an image,
    and rendering the corresponding OCR-extracted word as an image of the same size.
    """
    def __init__(self, 
                 pdf_path: str, 
                 page_number: int, 
                 bounding_box: list, 
                 word: str, 
                 denoise: bool = False,
                 font_params: dict = None,
                 edge_map_method: str = 'sobel',
                 debug: bool = False):
        """
        Initialize the WordImageProcessor with PDF details and word information.
        
        Args:
            pdf_path: Path to the PDF file.
            page_number: 1-based page number.
            bounding_box: A flat list of floats [x, y, width, height] defining the bounding box.
            word: The text to render.
            denoise: If True, denoise the extracted image.
            font_params: A dictionary of font parameters. Optional, defaults to None.
            edge_map_method: The method to use for edge detection.
            debug: If True, show the rendered image for debugging.
        """
        super().__init__()
        self.pdf_path = pdf_path
        self.page_number = page_number
        self.bounding_box = bounding_box
        self.word = word
        self.font_params = font_params
        self.denoise = denoise
        self.debug = debug
        self.edge_map_method = edge_map_method

        self.projection_bin_width = 4 # temporary

        # TODO check that font_params are ok

        ### Extract the word image from the PDF ###

        # get the extracted image
        self.extract_obj = WordImageExtractor(pdf_path=self.pdf_path, 
                                              page_number=self.page_number, 
                                              bounding_box=self.bounding_box)
        
        # denoise if requested
        if self.denoise:
            self.extract_obj.remove_specks()

        # get the extracted image
        self.set_extracted_image(self.extract_obj.get_modified_image())

        # get the centroid of the extracted image
        self.centroid = self._calculate_centroid(self.extracted_image)
        print(f"Centroid of extracted image: {self.centroid}")


        ### Render the word image ###

        # find the best font, font size, and smudge for the word (if not provided)
        if self.font_params is None:
            self.font_params = self._find_best_font_params()

        # create the rendered image
        self.render_obj = WordImageRenderer(word=self.word, 
                                            width=self.extract_obj.get_width(), 
                                            height=self.extract_obj.get_height(),
                                            font_size=self.font_params["fontsize"],
                                            typeface=self.font_params["typeface"],
                                            smudge=self.font_params["smudge"],
                                            centroid=self.centroid)

        # get the images
        self.set_rendered_image(self.render_obj.get_modified_image())

        self._validate_images()


    def _find_best_font_params(self):
        """
        Finds the best typeface, font size, and smudge for the word.

        Returns:

        """
        best_params = None
        best_score = 0
        for typeface in TYPEFACE_PATHS.keys():
            print(f"Testing typeface: {typeface}")
            for fontsize in range(int(self.extract_obj.get_height() - 10), int(self.extract_obj.get_height()), 1):
                for smudge in range(-5, 10, 1):
                    self.render_obj = WordImageRenderer(word=self.word, 
                                                        width=self.extract_obj.get_width(), 
                                                        height=self.extract_obj.get_height(),
                                                        font_size=fontsize,
                                                        typeface=typeface,
                                                        smudge=smudge,
                                                        centroid=self.centroid)
                    
                    self.set_rendered_image(self.render_obj.get_modified_image())

                    # compute the score
                    self.compute_similarity_scores()
                    similarity_score = self.get_mean_similarity_score()
                    
                    if similarity_score > best_score or best_params is None:
                        best_score = similarity_score
                        best_params = {"typeface": typeface, "fontsize": fontsize, "smudge": smudge}

        return best_params

    def set_extracted_image(self, extracted_image: np.ndarray):
        """
        Set the extracted image.
        """
        # convert to binary
        self.extracted_image = self._convert_to_binary(extracted_image)

        # Extract the edge maps
        self.extracted_edge_map = self._extract_edge_map(extracted_image, self.edge_map_method)

        # set the extracted image
        self.extracted_image = extracted_image

    def set_rendered_image(self, rendered_image: np.ndarray):
        """
        Set the rendered image.
        """
        # convert to binary
        self.rendered_image = self._convert_to_binary(rendered_image)

        # Extract the edge maps
        self.rendered_edge_map = self._extract_edge_map(rendered_image, self.edge_map_method)

        # set the rendered image
        self.rendered_image = rendered_image


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
        
        if self.extracted_image.shape != self.rendered_image.shape:
            raise ValueError("Extracted and rendered images have different dimensions.")
        
        if self.extracted_edge_map.shape != self.rendered_edge_map.shape:
            raise ValueError("Extracted and rendered edge maps have different dimensions.")

    def show_images(self):
        """
        Shows the extracted and rendered word images.
        """
        plt.imshow(self.extracted_image, cmap='gray')
        plt.title("Extracted Word Image")
        plt.axis('on')  # Keep the box
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.show()

        plt.imshow(self.extracted_edge_map, cmap='gray')
        plt.title("Extracted Word Edge Map")
        plt.axis('on')  # Keep the box
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.show()

        plt.imshow(self.rendered_image, cmap='gray')
        plt.title("Rendered Word Image")
        plt.axis('on')  # Keep the box
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.show()

        plt.imshow(self.rendered_edge_map, cmap='gray')
        plt.title("Rendered Word Edge Map")
        plt.axis('on')  # Keep the box
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.show()
        
    def compute_similarity_scores(self):
        """
        Computes the similarity scores for the word.
        """
        # Metrics on original image
        self.projection_similarity = self._projection_histogram_similarity(self.extracted_image, self.rendered_image, self.projection_bin_width)
        self.hu_similarity = self._hu_similarity(self.extracted_image, self.rendered_image)
        self.jaccard_similarity = self._jaccard_similarity(self.extracted_image, self.rendered_image)

        # Metrics on extracted edges
        self.chamfer_similarity = self._robust_chamfer_similarity(self.extracted_edge_map, self.rendered_edge_map)
        self.chamfer_dispersion_similarity = min(self._chamfer_dispersion_similarity(self.extracted_edge_map, self.rendered_edge_map),
                                                self._chamfer_dispersion_similarity(self.rendered_edge_map, self.extracted_edge_map))

    def get_similarity_scores(self) -> dict:
        """
        Returns the similarity scores for the word as a dictionary.

        Returns:
            dict: A dictionary containing the similarity scores.
        """
        if self.projection_similarity is None:
            self.compute_similarity_scores()

        return {"projection_similarity": self.projection_similarity, 
                "hu_similarity": self.hu_similarity, 
                "jaccard_similarity": self.jaccard_similarity, 
                "chamfer_similarity": self.chamfer_similarity,
                "chamfer_dispersion_similarity": self.chamfer_dispersion_similarity}

    def get_mean_similarity_score(self) -> float:
        """
        Returns the mean similarity score for the word.
        """
        return float(np.mean(list(self.get_similarity_scores().values())))


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

