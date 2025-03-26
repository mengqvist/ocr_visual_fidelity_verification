"""Visualization utilities for OCR verification results using new JSON parsing format.

This module provides tools for visualizing OCR results with quality indicators.
It now relies on the JSONParser from json_parser.py, which uses dataclasses to represent
the new JSON format (flat lists for polygons, etc.).

Example:
    visualizer = OCRVisualizer("doc.pdf", "ocr.json")
    visualizer.render_all_pages("output/", highlight_mode="confidence")
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import fitz  # PyMuPDF for PDF processing

# Import the new JSON parser
from src.parsers.json_parser import JSONParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRViewer:
    """Visualizes OCR results using the new JSON parsing format.

    This class loads a PDF and its associated OCR JSON (parsed via JSONParser),
    and renders pages with overlays based on OCR quality (confidence).
    """
    def __init__(self, pdf_path: str, json_path: str):
        """
        Args:
            pdf_path: Path to the original PDF file.
            json_path: Path to the OCR JSON file.
        """
        self.pdf_path = pdf_path
        self.json_path = json_path

        # Load PDF using PyMuPDF
        try:
            self.pdf_doc = fitz.open(self.pdf_path)
            logger.info(f"Loaded PDF with {len(self.pdf_doc)} pages")
        except Exception as e:
            logger.error(f"Failed to load PDF {self.pdf_path}: {e}")
            self.pdf_doc = None

        # Load OCR data using the new JSONParser
        try:
            self.ocr_parser = JSONParser(self.json_path)
            logger.info(f"Loaded OCR data with {len(self.ocr_parser.pages)} pages")
        except Exception as e:
            logger.error(f"Failed to load OCR data from {self.json_path}: {e}")
            self.ocr_parser = None

        # Create colormap for confidence scores using colorblind-friendly colors.
        self.confidence_cmap = LinearSegmentedColormap.from_list(
            'confidence',
            [(0, '#D55E00'),    # Dark orange for low confidence
             (0.5, '#56B4E9'),  # Light blue for medium confidence  
             (1, '#009E73')]    # Teal green for high confidence
        )

    def render_page_with_overlay(self, page_number: int, output_path: str, 
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
        if not self.pdf_doc or not self.ocr_parser:
            logger.error("PDF or OCR data not loaded")
            return False

        try:
            if page_number < 1 or page_number > len(self.pdf_doc):
                logger.error(f"Invalid page number: {page_number}")
                return False

            # Get PDF page and render it as an image
            pdf_page = self.pdf_doc[page_number - 1]
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Get OCR page (as parsed by JSONParser)
            ocr_page = self.ocr_parser.get_page(page_number)
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

                # For confidence mode, highlight words with low confidence
                if highlight_mode == 'confidence':
                    confidence = word.confidence
                    if confidence < confidence_threshold:
                        color = self.confidence_cmap(confidence)
                        alpha = 0.5
                    else:
                        # Skip high-confidence words for cleaner visualization
                        continue
                else:
                    # Default color for any other mode
                    color = 'blue'
                    alpha = 0.3

                rect = patches.Rectangle(
                    (x, y), width, height,
                    linewidth=1, edgecolor=color, facecolor=color, alpha=alpha
                )
                ax.add_patch(rect)

            plt.tight_layout()
            plt.savefig(output_path, dpi=dpi)
            plt.close(fig)
            logger.info(f"Saved visualization to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error rendering page {page_number}: {e}")
            return False

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

        Returns:
            List of paths to output images.
        """
        if not self.pdf_doc or not self.ocr_parser:
            logger.error("PDF or OCR data not loaded")
            return []

        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for page_num in range(1, len(self.pdf_doc) + 1):
            output_path = os.path.join(output_dir, f"page_{page_num:03d}.png")
            success = self.render_page_with_overlay(
                page_num, output_path, highlight_mode, confidence_threshold, dpi
            )
            if success:
                output_paths.append(output_path)
        return output_paths

    def create_quality_heatmap(self, page_number: int, output_path: str,
                               resolution: tuple = (1000, 1500)) -> bool:
        """
        Create a heatmap visualization of OCR quality for a given page.

        Args:
            page_number: Page number to visualize (1-based).
            output_path: Path to save the heatmap image.
            resolution: Resolution of the heatmap as (width, height).

        Returns:
            True if successful, False otherwise.
        """
        if not self.ocr_parser:
            logger.error("OCR data not loaded")
            return False

        try:
            ocr_page = self.ocr_parser.get_page(page_number)
            if not ocr_page:
                logger.error(f"OCR data not found for page {page_number}")
                return False

            width_res, height_res = resolution
            # Create an empty heatmap array; initialize with ones (high confidence)
            heatmap = np.ones((height_res, width_res), dtype=float)

            # Scale factors from OCR coordinates to heatmap resolution
            scale_x = width_res / ocr_page.width if ocr_page.width else 1
            scale_y = height_res / ocr_page.height if ocr_page.height else 1

            for word in ocr_page.words:
                if not word.bounding_box:
                    continue
                # Get scaled bounding box coordinates
                x_min = int(word.bounding_box.x * scale_x)
                y_min = int(word.bounding_box.y * scale_y)
                x_max = int((word.bounding_box.x + word.bounding_box.width) * scale_x)
                y_max = int((word.bounding_box.y + word.bounding_box.height) * scale_y)

                # Clip coordinates to heatmap dimensions
                x_min = max(0, min(x_min, width_res - 1))
                y_min = max(0, min(y_min, height_res - 1))
                x_max = max(0, min(x_max, width_res - 1))
                y_max = max(0, min(y_max, height_res - 1))

                # Fill the region with the word's confidence
                heatmap[y_min:y_max+1, x_min:x_max+1] = word.confidence

            fig, ax = plt.subplots(figsize=(10, 15))
            im = ax.imshow(heatmap, cmap=self.confidence_cmap, vmin=0, vmax=1)
            ax.axis('off')
            plt.title(f"OCR Quality Heatmap - Page {page_number}")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('OCR Confidence')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved heatmap to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating heatmap for page {page_number}: {e}")
            return False


def create_overlay_visualization(pdf_path: str, json_path: str, output_dir: str,
                                 highlight_mode: str = 'confidence') -> list:
    """
    Convenience function to create overlay visualizations for all pages.

    Args:
        pdf_path: Path to the PDF file.
        json_path: Path to the OCR JSON file.
        output_dir: Directory to save output images.
        highlight_mode: Mode for highlights ('confidence').

    Returns:
        List of paths to output images.
    """
    visualizer = OCRViewer(pdf_path, json_path)
    return visualizer.render_all_pages(output_dir, highlight_mode)


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
    parser.add_argument('--heatmap', action='store_true', help='Create heatmap visualization')

    args = parser.parse_args()

    visualizer = OCRViewer(args.pdf, args.json)
    os.makedirs(args.output, exist_ok=True)

    if args.page:
        output_path = os.path.join(args.output, f"page_{args.page:03d}.png")
        visualizer.render_page_with_overlay(args.page, output_path, args.mode, confidence_threshold=args.threshold)
        if args.heatmap:
            heatmap_path = os.path.join(args.output, f"heatmap_page_{args.page:03d}.png")
            visualizer.create_quality_heatmap(args.page, heatmap_path)
    else:
        visualizer.render_all_pages(args.output, args.mode, confidence_threshold=args.threshold)
        if args.heatmap:
            for page_num in range(1, len(visualizer.pdf_doc) + 1):
                heatmap_path = os.path.join(args.output, f"heatmap_page_{page_num:03d}.png")
                visualizer.create_quality_heatmap(page_num, heatmap_path)
