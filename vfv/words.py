from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from vfv import parsers
from PIL import Image, ImageDraw, ImageFont
import os
import json
from dotenv import load_dotenv, find_dotenv
from vfv.algorithms import ImageSimilarityAlgorithms, ImageColorAlgorithms, ImageDegradationAlgorithms
from config import TYPEFACE_PATHS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        return self.image

    def set_modified_image(self, image: np.ndarray):
        """
        Set a modified image.

        Args:
            image (np.ndarray): The image to set.
        """
        # convert to binary, if not already
        self.modified_image = image

    def get_modified_image(self) -> np.ndarray:
        """
        Get a modified image.

        Returns:
            np.ndarray: The modified image.
        """
        return self.modified_image
    

class WordImageCleaner(WordImage):
    """
    A class for processing an extracted word image from a PDF document.
    """
    def __init__(self, image: np.ndarray):
        """
        Initialize the WordImageCleaner with the image to process.

        Args:
            image (np.ndarray): The image to process.
        """
        super().__init__()

        # Apply binary thresholding to clean up fuzzy edges and prepare for further processing
        binary = self._convert_to_binary(image)

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
    A class for rendering a word as an image of a given size.
    """
    def __init__(self, word: str, width: int, height: int, font_size: int, typeface: str, smudge_distance: float=0, centroid: tuple[float, float]=None):
        """
        Initialize the WordImageRenderer with word and polygon details.

        Args:
            word (str): The text to render.
            width (int): The width of the image, in pixels.
            height (int): The height of the image, in pixels.
            font_size (int): The font size to use for the rendered image.
            typeface (str): The typeface to use for the rendered image.
            smudge_distance (float): The amount to smudge the image, in pixels.
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
        self.smudge_distance = smudge_distance

        if not isinstance(self.smudge_distance, (int, float)):
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

    def _get_available_typefaces(self) -> list[str]:
        """
        Get the available typefaces.

        Returns:
            list[str]: A list of the available fonts.
        """
        return list(TYPEFACE_PATHS.keys())

    def _load_font(self):
        """
        Load a TrueType font.

        Returns:
            ImageFont: The loaded font.
        """
        try:
            return ImageFont.truetype(TYPEFACE_PATHS[self.typeface], self.font_size)
        except IOError:
            raise IOError(f"Font {self.typeface} not found.")
        
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
                
        # Step 2: Render the text.
        font = self._load_font()
        canvas_center = (canvas_width // 2, canvas_height // 2)
        rendered_img = Image.new("RGB", (canvas_width, canvas_height), color='white')
        draw = ImageDraw.Draw(rendered_img)
        draw.text(canvas_center, self.word, fill='black', font=font, anchor='mm')
        rendered_np = self._convert_to_binary(np.array(rendered_img))

        # Save a copy of the unmodified rendered image.
        unmodified_np = rendered_np.copy()

        # apply smudge or fade to simulate text degradation, if requested
        if self.smudge_distance > 0:
            degraded_np = self._smudge(rendered_np, self.smudge_distance)
        elif self.smudge_distance < 0:
            degraded_np = self._fade(rendered_np, self.smudge_distance)
        else:
            degraded_np = self._convert_to_binary(rendered_np)

        # Step 3: Calculate the center of mass from the rendered image.
        rendered_x, rendered_y = self._calculate_centroid(degraded_np)  # returns (x, y)

        # Step 4: Compute crop coordinates so that the rendered centroid aligns with self.centroid.
        if self.centroid is None:
            target_x, target_y = canvas_width // 2, canvas_height // 2
        else:
            target_x, target_y = self.centroid
        
        # Determine top-left corner of crop: we want rx to end up at target_x, and ry at target_y.
        crop_x = int(rendered_x - target_x)
        crop_y = int(rendered_y - target_y)
        
        # Ensure the crop stays within the canvas boundaries.
        crop_x = max(0, min(crop_x, canvas_width - self.width))
        crop_y = max(0, min(crop_y, canvas_height - self.height))
            
        # Crop both versions using the same coordinates.
        cropped_unmodified = unmodified_np[crop_y:crop_y + self.height, crop_x:crop_x + self.width]
        cropped_degraded = degraded_np[crop_y:crop_y + self.height, crop_x:crop_x + self.width]

        # Ensure binary conversion if needed.
        clean = self._convert_to_binary(cropped_unmodified)
        degraded = self._convert_to_binary(cropped_degraded)
        
        # Save them separately:
        self.set_image(clean)                # unsmudged, original rendered image
        self.set_modified_image(degraded)      # degraded (smudged/faded) image 

    def smudge(self, distance_threshold: float = 1.5):
        """
        Apply a distance-based smudging for natural-looking text degradation.
        
        Args:
            distance_threshold: Distance threshold for smudging (in pixels)
                            Higher values create more smudging
        """
        output = self._smudge(self.get_image().copy(), distance_threshold)
        self.set_modified_image(output)

    def fade(self, distance_threshold: float = 1.5):
        """
        Apply a distance-based fade for more natural text degradation.

        Args:
            distance_threshold: Distance threshold for fade (in pixels)
                            Higher values create more fade

        """
        output = self._fade(self.get_image().copy(), distance_threshold)
        self.set_modified_image(output)

    def degrade(self, distance_threshold: float):
        """
        Apply a distance-based smudging and fading for natural-looking text degradation.

        Args:
            distance_threshold: Distance threshold for degradation (in pixels)
                            Positive values create more smudging, negative values create more fading
        """
        # apply smudge or fade to simulate text degradation, if requested
        if distance_threshold > 0:
            output = self._smudge(self.get_image().copy(), distance_threshold)
        elif distance_threshold < 0:
            output = self._fade(self.get_image().copy(), distance_threshold)
        elif distance_threshold == 0:
            output = self.get_image().copy()

        self.set_modified_image(output)


class WordPairProcessor(ImageSimilarityAlgorithms):
    """
    A wrapper class for processing extracting a single word from a PDF document as an image,
    and rendering the corresponding OCR-extracted word as an image of the same size.
    """
    def __init__(self, 
                 pdf_parser: parsers.PDFParser, 
                 json_parser: parsers.JSONParser, 
                 page_number: int, 
                 word_index: int, 
                 denoise: bool = True,
                 font_params: dict = None,
                 edge_map_method: str = 'sobel',
                 debug: bool = False):
        """
        Initialize the WordImageProcessor with PDF details and word information.
        
        Args:
            pdf_parser: The PDF parser.
            json_parser: The JSON parser.
            page_number: 1-based page number.
            word_index: 0-based word index.
            denoise: If True, denoise the extracted image.
            font_params: A dictionary of font parameters. Optional, defaults to None.
            edge_map_method: The method to use for edge detection.
            debug: If True, show the rendered image for debugging.
        """
        super().__init__()
        self.page_number = page_number
        self.word_index = word_index
        self.word = json_parser.get_page(self.page_number).get_word(self.word_index).get_content()
        self.bounding_box = json_parser.get_page(self.page_number).get_word(self.word_index).get_bounding_box()

        self.font_params = font_params
        self.denoise = denoise
        self.debug = debug
        self.edge_map_method = edge_map_method

        self.projection_bin_width = 4 # temporary

        # TODO check that font_params are ok

        ### Extract the word image from the PDF ###

        # get the extracted image
        self.extract_obj = WordImageCleaner(image=pdf_parser.get_word_image(self.page_number, self.word_index))
        
        # denoise if requested
        if self.denoise:
            self.extract_obj.remove_specks()

        # get the extracted image
        self.set_extracted_image(self.extract_obj.get_modified_image())

        # get the centroid of the extracted image
        self.centroid = self._calculate_centroid(self.extracted_image)


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
                                            smudge_distance=self.font_params["degradation_threshold"],
                                            centroid=self.centroid)

        # get the images
        self.set_unmodified_rendered_image(self.render_obj.get_image())
        self.set_rendered_image(self.render_obj.get_modified_image())

        self._validate_images()

    def _find_best_font_params(self):
        """
        Finds the best typeface, font size, and smudge for the word.

        Returns:

        """
        best_params = None
        best_score = -1
        width = self.extract_obj.get_width()
        height = self.extract_obj.get_height()
        for typeface in TYPEFACE_PATHS.keys():
            for fontsize in range(int(height - 40), int(height), 2):
                # Render the word once with no degradation.
                temp_renderer = WordImageRenderer(
                    word=self.word,
                    width=width,
                    height=height,
                    font_size=fontsize,
                    typeface=typeface,
                    smudge_distance=0,
                    centroid=self.centroid
                )
                # Cache the baseline rendered image.
                base_rendered = self._convert_to_binary(temp_renderer.get_image().copy())

                # discard if the baseline image has no white border
                if best_score != -1 and not self._has_white_border(base_rendered):
                    continue
                
                # Loop over degradation thresholds.
                for degradation_threshold in range(-2, 9):
                    # Apply degradation on the baseline image directly.
                    if degradation_threshold > 0:
                        rendered_variant = self._smudge(base_rendered, degradation_threshold)
                    elif degradation_threshold < 0:
                        rendered_variant = self._fade(base_rendered, degradation_threshold)
                    else:
                        rendered_variant = base_rendered

                    # Compute similarity score using the local helper.
                    scores = self._compute_similarity_for_variant(rendered_variant)
                    similarity_score = np.mean(list(scores.values()))

                    # Update best parameters if the score is higher.
                    if similarity_score > best_score or best_params is None:
                        best_score = similarity_score
                        best_params = {
                            "typeface": typeface,
                            "fontsize": fontsize,
                            "degradation_threshold": degradation_threshold
                        }

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

    def set_unmodified_rendered_image(self, unmodified_rendered_image: np.ndarray):
        """
        Set the unmodified rendered image.
        """
        self.unmodified_rendered_image = self._convert_to_binary(unmodified_rendered_image)

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
            raise ValueError(f"Extracted and rendered images have different dimensions: {self.extract_obj.get_dimensions()} != {self.render_obj.get_dimensions()}")
        
        if self.extract_obj.get_modified_dimensions() != self.render_obj.get_modified_dimensions():
            raise ValueError(f"Extracted and rendered (modified) images have different dimensions: {self.extract_obj.get_modified_dimensions()} != {self.render_obj.get_modified_dimensions()}")
        
        if self.extracted_image.shape != self.rendered_image.shape:
            raise ValueError(f"Extracted and rendered images have different dimensions: {self.extracted_image.shape} != {self.rendered_image.shape}")
        
        if self.extracted_edge_map.shape != self.rendered_edge_map.shape:
            raise ValueError(f"Extracted and rendered edge maps have different dimensions: {self.extracted_edge_map.shape} != {self.rendered_edge_map.shape}")

    def show_images(self, show_edge_maps: bool = False):
        """
        Shows the extracted and rendered word images.
        """
        plt.imshow(self.extracted_image, cmap='gray')
        plt.title("Extracted Word Image")
        plt.axis('on')  # Keep the box
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.show()

        if show_edge_maps:
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

        if show_edge_maps:
            plt.imshow(self.rendered_edge_map, cmap='gray')
            plt.title("Rendered Word Edge Map")
            plt.axis('on')  # Keep the box
            plt.xticks([])  # Remove x-axis ticks
            plt.yticks([])  # Remove y-axis ticks
            plt.show() 

    def _compute_similarity_for_variant(self, rendered_variant: np.ndarray) -> float:
        """
        Computes the similarity score for a given rendered variant.
        """
        rendered_variant_edge_map = self._extract_edge_map(rendered_variant, self.edge_map_method)
        if not self._has_white_border(rendered_variant):
            scores = {"projection_similarity": 0.0,
                     "hu_similarity": 0.0,
                     "jaccard_similarity": 0.0,
                     "chamfer_similarity": 0.0,
                     "black_pixel_similarity": 0.0}
        else:
            scores = self._compute_similarity(image1=self.extracted_image, 
                                             image2=rendered_variant, 
                                             image1_edge_map=self.extracted_edge_map, 
                                             image2_edge_map=rendered_variant_edge_map)
        return scores

    def compute_similarity_scores(self):
        """
        Computes the similarity scores for the word.
        """
        scores = self._compute_similarity(image1=self.extracted_image, 
                                         image2=self.rendered_image, 
                                         image1_edge_map=self.extracted_edge_map, 
                                         image2_edge_map=self.rendered_edge_map)
        self.projection_similarity = scores["projection_similarity"]
        self.hu_similarity = scores["hu_similarity"]
        self.jaccard_similarity = scores["jaccard_similarity"]
        self.chamfer_similarity = scores["chamfer_similarity"]
        self.black_pixel_similarity = scores["black_pixel_similarity"]

    def get_similarity_scores(self) -> dict:
        """
        Returns the similarity scores for the word as a dictionary.

        Returns:
            dict: A dictionary containing the similarity scores.
        """
        return {"projection_similarity": self.projection_similarity, 
                "hu_similarity": self.hu_similarity, 
                "jaccard_similarity": self.jaccard_similarity, 
                "chamfer_similarity": self.chamfer_similarity,
                "black_pixel_similarity": self.black_pixel_similarity}

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
        A class for processing a document to obtain quality scores for each word.
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
        self.json_loader = parsers.JSONParser(json_path)
        self.pdf_loader = parsers.PDFParser(pdf_path, self.json_loader)

        # process the document to obtain quality scores for each word
        self._process_document()

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


    def get_json_loader(self) -> parsers.JSONParser:
        """
        Get the JSON loader.

        Returns:
            json_parser.JSONParser: The JSON loader.
        """
        return self.json_loader
    
    def get_pdf_loader(self) -> parsers.PDFParser:
        """
        Get the PDF loader.
        """
        return self.pdf_loader

    def _process_document(self):
        """
        Process the document to obtain quality scores for each word.

        Returns:
            list: A list of WordPairProcessor objects.
        """
        words = {}
        num_pages = len(self.json_loader.pages)
        for page_number in range(1, num_pages + 1):
            logger.info(f"Processing page {page_number} of {num_pages}")
            page_obj = self.json_loader.get_page(page_number)
            for i, word in enumerate(page_obj.get_words()):
                logger.info(f"Processing word {i} of {len(page_obj.get_words())} on page {page_number}")
                word_processor = WordPairProcessor(pdf_parser=self.pdf_loader,
                                                   json_parser=self.json_loader,
                                                   page_number=page_number, 
                                                   word_index=i,
                                                   denoise=True,
                                                   font_params=None,
                                                   edge_map_method='sobel',
                                                   debug=False)
                word_processor.compute_similarity_scores()

                if words.get(page_number) is None:
                    words[page_number] = []

                words[page_number].append(word_processor)
    
        self.words = words

    def get_page(self, page_number: int) -> list[WordPairProcessor]:
        """
        Get the words for a given page.
        """
        return self.words[page_number]
    
    def export_json(self, output_path: str):
        """
        Export the document to a JSON file.

        Each page is stored with an array of words, where each word is saved with its text content,
        bounding box (as [x, y, width, height]), quality score, and detailed similarity scores.

        Args:
            output_path (str): The file path where the JSON will be saved.
        """
        export_data = {"pages": {}}

        # Iterate over each page in the document
        for page_number, word_processors in self.words.items():
            # Use string keys for pages to mimic typical JSON formats.
            export_data["pages"][str(page_number)] = []

            # Iterate over each word processor for this page.
            for word_index, wp in enumerate(word_processors):
                word_info = {
                    "word_index": word_index,
                    "text": wp.word,
                    # Convert bounding_box to a list [x, y, width, height] (if available).
                    "bounding_box": wp.bounding_box if wp.bounding_box else None,
                    "quality_score": wp.get_mean_similarity_score(),
                    "similarity_scores": wp.get_similarity_scores()
                }
                export_data["pages"][str(page_number)].append(word_info)

        # Write the JSON file.
        with open(output_path, "w") as outfile:
            json.dump(export_data, outfile, indent=2)
        logger.info(f"Exported quality data to JSON file at: {output_path}")
