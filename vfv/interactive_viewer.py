import os
import fitz  # PyMuPDF for PDF processing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from PIL import Image

# Use relative import since interactive_viewer.py is in src/gui/
from src.parsers.json_parser import JSONParser
from src.gui.viewer import OCRViewer  # our updated visualizer code

class InteractiveOCRViewer:
    def __init__(self, pdf_path, json_path, highlight_mode='confidence',
                 confidence_threshold=1.0, dpi=150):
        """
        Initialize the interactive viewer.
        
        Args:
            pdf_path (str): Path to the PDF file.
            json_path (str): Path to the OCR JSON file.
            highlight_mode (str): Currently only 'confidence' is supported.
            confidence_threshold (float): Highlight words below this confidence.
            dpi (int): DPI for rendering the PDF page.
        """
        self.vis = OCRViewer(pdf_path, json_path)
        self.highlight_mode = highlight_mode
        self.confidence_threshold = confidence_threshold
        self.dpi = dpi
        self.current_page = 1

        # Create a figure with two side-by-side subplots.
        self.fig, (self.ax_img, self.ax_text) = plt.subplots(1, 2, figsize=(24, 12))
        plt.subplots_adjust(bottom=0.2)

        # Connect resize event to update drawing.
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)

        # Create navigation buttons.
        self.axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
        self.axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.bprev = Button(self.axprev, 'Previous')
        self.bnext = Button(self.axnext, 'Next')
        self.bprev.on_clicked(self.prev_page)
        self.bnext.on_clicked(self.next_page)

        self.draw_page()

    def on_resize(self, event):
        """Redraw the page when the window is resized."""
        self.draw_page()

    def adjust_font_size(self, ax, text_str, init_fontsize, bbox_width, bbox_height):
        """
        Iteratively adjust the font size so that the rendered text fits within the desired bounding box.
        
        Args:
            ax: The axis to render the text.
            text_str (str): The text string to measure.
            init_fontsize (float): Initial guess for the font size (in points).
            bbox_width (float): Desired maximum width in display pixels.
            bbox_height (float): Desired maximum height in display pixels.
        
        Returns:
            The maximum font size (in points) that causes the text to fit inside the box.
        """
        # Create a dummy text artist at an arbitrary position.
        text_artist = ax.text(0, 0, text_str, fontsize=init_fontsize,
                              ha='center', va='center')
        renderer = self.fig.canvas.get_renderer()
        extent = text_artist.get_window_extent(renderer=renderer)
        font_size = init_fontsize
        # Reduce font size until the text extent fits in the desired box.
        while (extent.width > bbox_width or extent.height > bbox_height) and font_size > 1:
            font_size *= 0.95  # reduce by 5%
            text_artist.set_fontsize(font_size)
            extent = text_artist.get_window_extent(renderer=renderer)
        text_artist.remove()
        return font_size

    def draw_page(self):
        """Draw the current page in both subplots."""
        # Clear both axes.
        self.ax_img.clear()
        self.ax_text.clear()

        # Render PDF page to an image.
        pdf_page = self.vis.pdf_doc[self.current_page - 1]
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        self.ax_img.imshow(np.array(img))
        self.ax_img.axis('off')
        self.ax_img.set_title(f"Page {self.current_page} - Scanned with Overlays")

        # Get OCR page data.
        ocr_page = self.vis.ocr_parser.get_page(self.current_page)
        if ocr_page is not None:
            # Compute scale factors from OCR coordinate space to image pixels.
            scale_x = pix.width / ocr_page.width if ocr_page.width else 1
            scale_y = pix.height / ocr_page.height if ocr_page.height else 1

            # Overlay bounding boxes on the left image for low-confidence words.
            for word in ocr_page.words:
                if not word.bounding_box:
                    continue
                if self.highlight_mode == 'confidence':
                    conf = word.confidence
                    if conf < self.confidence_threshold:
                        color = self.vis.confidence_cmap(conf)
                        alpha = 0.5
                    else:
                        continue  # skip high-confidence words
                else:
                    color = 'blue'
                    alpha = 0.3

                bb = word.bounding_box
                x = bb.x * scale_x
                y = bb.y * scale_y
                width = bb.width * scale_x
                height = bb.height * scale_y
                rect = patches.Rectangle(
                    (x, y), width, height,
                    linewidth=1, edgecolor=color, facecolor=color, alpha=alpha
                )
                self.ax_img.add_patch(rect)

            # Set the right subplot to use the same coordinate system as the scanned image.
            self.ax_text.set_xlim(0, pix.width)
            self.ax_text.set_ylim(pix.height, 0)  # origin at top-left
            self.ax_text.set_title("OCR Text Rendered in Position")
            self.ax_text.axis('off')
            
            # Render each word centered inside its bounding box.
            for word in ocr_page.words:
                if not word.bounding_box:
                    continue
                bb = word.bounding_box
                # Center position for the word in display (image) coordinates.
                x_center = (bb.x + bb.width/2) * scale_x
                y_center = (bb.y + bb.height/2) * scale_y

                # Desired bounding box dimensions in display pixels.
                desired_width = bb.width * scale_x
                desired_height = bb.height * scale_y
                # Initial guess: 95% of the height in points.
                init_fontsize = 0.95 * bb.height * scale_y * (72 / self.dpi)
                # Adjust font size to fit inside both width and height.
                fontsize = self.adjust_font_size(self.ax_text, word.content, init_fontsize,
                                                 desired_width, desired_height)
                # Enforce a minimum font size (e.g., 2pt) if needed.
                fontsize = max(fontsize, 2)

                self.ax_text.text(x_center, y_center, word.content,
                                  fontsize=fontsize,
                                  verticalalignment='center',
                                  horizontalalignment='center',
                                  color='black')
                
                # Optionally, draw a filled rectangle behind the text as a quality indicator.
                if self.highlight_mode == 'confidence' and word.confidence < self.confidence_threshold:
                    rect = patches.Rectangle(
                        ((bb.x) * scale_x, (bb.y) * scale_y), desired_width, desired_height,
                        linewidth=0.1, facecolor=self.vis.confidence_cmap(word.confidence),
                        alpha=0.3
                    )
                    self.ax_text.add_patch(rect)
        self.fig.canvas.draw_idle()

    def next_page(self, event):
        total_pages = len(self.vis.pdf_doc)
        self.current_page = self.current_page + 1 if self.current_page < total_pages else 1
        self.draw_page()

    def prev_page(self, event):
        total_pages = len(self.vis.pdf_doc)
        self.current_page = self.current_page - 1 if self.current_page > 1 else total_pages
        self.draw_page()

def interactive_viewer(pdf_path, json_path, highlight_mode='confidence',
                       confidence_threshold=1.0, dpi=150):
    viewer = InteractiveOCRViewer(pdf_path, json_path, highlight_mode, confidence_threshold, dpi)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive OCR Viewer")
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--json', required=True, help='Path to OCR JSON file')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for rendering')
    parser.add_argument('--threshold', type=float, default=1.0, help='Confidence threshold')
    args = parser.parse_args()
    interactive_viewer(args.pdf, args.json, dpi=args.dpi, confidence_threshold=args.threshold)
