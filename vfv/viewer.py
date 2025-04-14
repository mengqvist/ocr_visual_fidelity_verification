import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
import fitz  # PyMuPDF for PDF processing
from vfv.words import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRViewer:
    """Visualizes OCR results using the new JSON parsing format.

    This class loads a PDF and its associated OCR JSON (parsed via JSONParser),
    and renders pages with overlays based on OCR quality (confidence).
    """
    def __init__(self, pdf_path: str, json_path: str, dpi: int = 72):
        """
        Args:
            pdf_path: Path to the original PDF file.
            json_path: Path to the OCR JSON file.   
            dpi: DPI for rendering.
        """
        self.pdf_path = pdf_path
        self.json_path = json_path
        self.dpi = dpi

        self.pages = None
        self.pixmaps = None

        # load pdf and json
        self.document = Document(pdf_path=pdf_path, json_path=json_path)

        # Create colormap for confidence scores using colorblind-friendly colors.
        self.confidence_cmap = LinearSegmentedColormap.from_list(
            'confidence',
            [(0, '#D55E00'),    # Dark orange for low confidence
             (0.5, '#56B4E9'),  # Light blue for medium confidence  
             (1, '#009E73')]    # Teal green for high confidence
        )

        # load pages and pixmaps
        self._load_pages()

    def _load_pages(self):
        """
        Load a page from the PDF and return a fitz.Pixmap object.
        """
        if not self.document:
            logger.error("PDF or OCR data not loaded")
            return False
        
        self.pdf_pages = []
        self.pixmaps = []

        for page_number in range(1, len(self.document.get_json_loader().pages) + 1):
            # Get PDF page and render it as an image
            self.pdf_pages.append(fitz.open(self.pdf_path)[page_number - 1])
            self.pixmaps.append(self.pdf_pages[-1].get_pixmap(matrix=fitz.Matrix(150/self.dpi, 150/self.dpi)))

    def _add_quality_rectangle(self, ax, x: float, y: float, width: float, height: float, 
                            quality: float, highlight_mode: str, 
                            confidence_threshold: float = 1.0):
        """
        Adds a rectangle overlay to the given axis based on the quality score.

        Args:
            ax: The matplotlib axes object on which to add the rectangle.
            x: The x-coordinate for the rectangle.
            y: The y-coordinate for the rectangle.
            width: The width of the rectangle.
            height: The height of the rectangle.
            quality: The quality/confidence score associated with the text.
            highlight_mode: The mode for highlighting (e.g., 'confidence').
            confidence_threshold: The threshold below which the overlay is applied.
            
        Returns:
            The matplotlib axes with the rectangle overlay added.
        """
        if highlight_mode == 'confidence':
            if quality < confidence_threshold:
                color = self.confidence_cmap(quality)
                alpha = 0.5
            else:
                # Do not add an overlay if above threshold
                return ax
        else:
            color = 'blue'
            alpha = 0.3

        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        return ax

    def _add_quality_rectangle_pil(self,image: Image.Image, x: float, y: float, width: float, height: float, 
                                quality: float, highlight_mode: str, confidence_threshold: float = 1.0,) -> Image.Image:
        """
        Adds a quality rectangle overlay to a PIL image based on the quality score.
        
        Args:
            image: The base PIL image onto which the overlay will be applied.
            x: The x-coordinate of the rectangle.
            y: The y-coordinate of the rectangle.
            width: The width of the rectangle.
            height: The height of the rectangle.
            quality: The quality (or confidence) score.
            highlight_mode: Mode for highlighting (e.g., 'confidence').
            confidence_threshold: Threshold below which the overlay is applied.
        
        Returns:
            A new PIL image with the overlay applied.
        """
        # Convert the base image to RGBA if it's not already, so we can work with transparency.
        base_image = image.convert("RGBA")
        overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        
        if highlight_mode == 'confidence':
            if quality < confidence_threshold:
                # Use the colormap to get a color, if provided. The colormap returns normalized floats in 0-1.
                norm_color = self.confidence_cmap(quality)
                # Convert normalized values to integers in the range 0-255.
                color = tuple(int(c * 255) for c in norm_color[:3])

                alpha = 128  # Semi-transparent
            else:
                # No overlay is added if the quality is above threshold.
                return image
        else:
            color = (0, 0, 255)
            alpha = 77

        fill_color = color + (alpha,)
        # Draw the rectangle on the overlay.
        draw.rectangle([int(x), int(y), int(x + width), int(y + height)], fill=fill_color, outline=fill_color)
        
        # Composite the overlay with the base image.
        combined = Image.alpha_composite(base_image, overlay)
        return combined.convert("RGB")

    def _render_original_page(self, page_number: int, output_path: str, 
                                 highlight_mode: str = 'confidence',
                                 confidence_threshold: float = 1.0,
                                 dpi: int = 150) -> bool:
        """
        Render a PDF page with OCR overlay highlighting words based on confidence.

        Args:
            page_number: Page number to render (1-based).
            output_path: Path to save the output image.
            highlight_mode: Mode for highlights ('confidence').
            confidence_threshold: Confidence threshold below which words are highlighted.
            dpi: DPI for rendering.

        Returns:
            True if successful, False otherwise.
        """
        pix = self.pixmaps[page_number - 1]
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Get OCR page (as parsed by JSONParser)
        ocr_page = self.document.get_json_loader().get_page(page_number)
        if not ocr_page:
            logger.error(f"OCR data not found for page {page_number}")
            return False

        # Compute scaling factors from OCR coordinates to image pixels
        scale_x = pix.width / ocr_page.width if ocr_page.width else 1
        scale_y = pix.height / ocr_page.height if ocr_page.height else 1

        # Create figure and display PDF image
        fig, ax = plt.subplots(figsize=(pix.width/100, pix.height/100))
        ax.imshow(np.array(img))
        ax.axis('off')
        ax.set_title(f"Page {page_number} - Scanned Image with OCR Overlays")

        # Iterate over words and draw bounding boxes
        for word in ocr_page.words:
            # Use the precomputed bounding_box (from the new JSON parsing)
            if not word.bounding_box:
                continue

            # Scale bounding box coordinates to image space
            x = word.bounding_box.x * scale_x
            y = word.bounding_box.y * scale_y
            width = word.bounding_box.width * scale_x
            height = word.bounding_box.height * scale_y

            # Add the quality rectangle
            ax = self._add_quality_rectangle(ax, x, y, width, height, word.confidence, highlight_mode, confidence_threshold)


        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved original overlay visualization to {output_path}")

    def _render_extracted_words(self, page_number: int, output_path: str,
                                highlight_mode: str = 'confidence',
                                confidence_threshold: float = 1.0,
                                dpi: int = 150, degradation: bool = True) -> bool:
        """
        Render the extracted words with simulated degradation on a blank page.

        Args:
            page_number: Page number to render (1-based).
            output_path: Path to save the output image.
            dpi: DPI for rendering.
            degradation: Whether to simulate degradation.
        """

        ### Render Image 2: Blank page with rendered words and similarity scores
        # Create a blank white canvas
        pix = self.pixmaps[page_number - 1]
        rendered_canvas = Image.new("RGB", (pix.width, pix.height), "white")
        draw = ImageDraw.Draw(rendered_canvas)

        # Get OCR page (as parsed by JSONParser)
        ocr_page = self.document.get_json_loader().get_page(page_number)
        if not ocr_page:
            logger.error(f"OCR data not found for page {page_number}")
            return False

        # Compute scaling factors from OCR coordinates to image pixels
        scale_x = pix.width / ocr_page.width if ocr_page.width else 1
        scale_y = pix.height / ocr_page.height if ocr_page.height else 1

        # Process each word on the page and overlay the rendered word image with similarity score
        for i, word_pair in enumerate(self.document.get_page(page_number)):

            # Scale bounding box coordinates
            bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height = word_pair.bounding_box
            x = int(bounding_box_x * scale_x)
            y = int(bounding_box_y * scale_y)
            width = int(bounding_box_width * scale_x)
            height = int(bounding_box_height * scale_y)

            # Convert numpy array to PIL Image
            if degradation:
                image = word_pair.render_obj.get_modified_image()
            else:
                image = word_pair.render_obj.get_image()
            image = Image.fromarray(image)

            # Convert to RGB if grayscale
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize the rendered word image to match the bounding box dimensions
            image = image.resize((width, height))

            # Paste using only the upper-left corner, since the image now has the correct size
            rendered_canvas.paste(image, (x, y))

            # Add the quality rectangle
            confidence = word_pair.get_mean_similarity_score()
            rendered_canvas = self._add_quality_rectangle_pil(rendered_canvas, x, y, width, height, confidence, highlight_mode, confidence_threshold)

        rendered_canvas.save(output_path)
        logger.info(f"Saved rendered words overlay visualization to {output_path}")

    def render_all_pages(self, output_dir: str, 
                         highlight_mode: str = 'confidence',
                         confidence_threshold: float = 1.0,
                         dpi: int = 150) -> list:
        """
        Render all PDF pages with overlays.

        Args:
            output_dir: Directory to save output images.
            highlight_mode: Mode for highlights ('confidence').
            confidence_threshold: Threshold for confidence highlighting.
            dpi: DPI for rendering.
        """

        os.makedirs(output_dir, exist_ok=True)

        for page_num in range(1, len(self.document.get_json_loader().pages) + 1):
            if page_num != 1:
                continue

            # Render original page
            output_path = os.path.join(output_dir, f"page_{page_num:03d}_original.png")
            self._render_original_page(
                page_num, output_path, highlight_mode, confidence_threshold, dpi
            )

            # Render extracted words
            output_path = os.path.join(output_dir, f"page_{page_num:03d}_rendered.png")
            self._render_extracted_words(
                page_num, output_path, highlight_mode, confidence_threshold, dpi, degradation=False
            )

            # Render degraded words
            output_path = os.path.join(output_dir, f"page_{page_num:03d}_rendered_degraded.png")
            self._render_extracted_words(
                page_num, output_path, highlight_mode, confidence_threshold, dpi, degradation=True
            )

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Visualize OCR verification results with new JSON format')
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--json', required=True, help='Path to OCR JSON file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=1.0, help='Confidence threshold')
    parser.add_argument('--mode', choices=['confidence'], default='confidence', help='Highlighting mode')
    parser.add_argument('--page', type=int, help='Specific page to visualize (optional)')

    args = parser.parse_args()

    visualizer = OCRViewer(args.pdf, args.json)
    os.makedirs(args.output, exist_ok=True)

    if args.page:
        output_path = os.path.join(args.output, f"page_{args.page:03d}.png")
        visualizer.render_page_with_overlay(args.page, output_path, args.mode, confidence_threshold=args.threshold)

    else:
        visualizer.render_all_pages(args.output, args.mode, confidence_threshold=args.threshold)

