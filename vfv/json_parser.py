import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Represents a bounding box for a text element."""
    x: float
    y: float
    width: float
    height: float
    
    @classmethod
    def from_polygon(cls, polygon: List[float]) -> 'BoundingBox':
        """Create a bounding box from a polygon."""
        if not polygon or len(polygon) < 4:
            return cls(x=0, y=0, width=0, height=0)
        
        # Extract x and y coordinates from flat list [x1,y1,x2,y2,...]
        x_values = polygon[0::2]  # Every even index (0, 2, 4, ...)
        y_values = polygon[1::2]  # Every odd index (1, 3, 5, ...)
        
        x = min(x_values)
        y = min(y_values)
        width = max(x_values) - x
        height = max(y_values) - y
        
        return cls(x=x, y=y, width=width, height=height)

@dataclass
class BoundingRegion:
    """Represents a bounding region with page number and polygon."""
    page_number: int
    polygon: List[float]
    bounding_box: BoundingBox = None
    
    def __post_init__(self):
        if self.bounding_box is None and self.polygon:
            self.bounding_box = BoundingBox.from_polygon(self.polygon)

@dataclass
class Span:
    """Represents a span of text with offset and length."""
    offset: int
    length: int

@dataclass
class OCRWord:
    """Represents a word from OCR output."""
    content: str
    confidence: float
    polygon: List[Dict[str, float]]
    span: Span
    bounding_box: BoundingBox = None
    
    def __post_init__(self):
        if self.bounding_box is None and self.polygon:
            self.bounding_box = BoundingBox.from_polygon(self.polygon)

@dataclass
class OCRLine:
    """Represents a line of text from OCR output."""
    content: str
    polygon: List[Dict[str, float]]
    spans: List[Span]
    words: List[OCRWord] = field(default_factory=list)
    bounding_box: BoundingBox = None
    
    def __post_init__(self):
        if self.bounding_box is None and self.polygon:
            self.bounding_box = BoundingBox.from_polygon(self.polygon)

@dataclass
class TableCell:
    """Represents a cell in a table."""
    content: str
    row_index: int
    column_index: int
    kind: str  # header, data, etc.
    bounding_regions: List[BoundingRegion]
    spans: List[Span]

@dataclass
class OCRTable:
    """Represents a table from OCR output."""
    row_count: int
    column_count: int
    cells: List[TableCell]
    bounding_regions: List[BoundingRegion]
    spans: List[Span]

@dataclass
class Figure:
    """Represents a figure or image in the document."""
    id: str
    bounding_regions: List[BoundingRegion]
    spans: List[Span]
    elements: List[str] = field(default_factory=list)

@dataclass
class KeyValuePair:
    """Represents a key-value pair in the document."""
    confidence: float
    key: Dict[str, Any]  # Contains content, boundingRegions, spans
    value: Dict[str, Any]  # Contains content, boundingRegions, spans

    @property
    def key_text(self) -> str:
        """Get the text content of the key."""
        return self.key.get('content', '')
    
    @property
    def value_text(self) -> str:
        """Get the text content of the value."""
        return self.value.get('content', '')
    
@dataclass
class Paragraph:
    """Represents a paragraph in the document."""
    content: str
    bounding_regions: List[BoundingRegion]
    spans: List[Span]

@dataclass
class Section:
    """Represents a section in the document."""
    spans: List[Span]
    elements: List[str] = field(default_factory=list)

@dataclass
class Style:
    """Represents a style in the document."""
    confidence: float
    is_handwritten: bool
    spans: List[Span]

@dataclass
class Language:
    """Represents a language detected in the document."""
    locale: str
    confidence: float
    spans: List[Span]

@dataclass
class OCRPage:
    """Represents a page from OCR output."""
    page_number: int
    angle: float
    width: float
    height: float
    unit: str
    words: List[OCRWord] = field(default_factory=list)
    lines: List[OCRLine] = field(default_factory=list)
    spans: List[Span] = field(default_factory=list)

class JSONParser:
    """Represents a complete OCR document from Azure Document Intelligence."""
    
    def __init__(self, filepath: str = None):
        """Initialize OCR document, optionally from a JSON file."""
        self.filepath = filepath
        self.filename = os.path.basename(filepath) if filepath else None
        
        # Document metadata
        self.status: str = ""
        self.created_date_time: str = ""
        self.last_updated_date_time: str = ""
        
        # Analyze result fields
        self.api_version: str = ""
        self.model_id: str = ""
        self.content: str = ""
        self.content_format: str = ""
        self.string_index_type: str = ""
        
        # Document structure
        self.pages: List[OCRPage] = []
        self.tables: List[OCRTable] = []
        self.paragraphs: List[Paragraph] = []
        self.key_value_pairs: List[KeyValuePair] = []
        self.figures: List[Figure] = []
        self.sections: List[Section] = []
        self.styles: List[Style] = []
        self.languages: List[Language] = []
        
        # Original data
        self.raw_data: Dict[str, Any] = {}
        
        if filepath:
            self.load_from_file(filepath)
        
    def __repr__(self) -> str:
        """Return a detailed string representation of the document."""
        return (
            f"(OCRDocument, "
            f"pages={len(self.pages)}, "
            f"tables={len(self.tables)}, "
            f"figures={len(self.figures)}, "
            f"key_value_pairs={len(self.key_value_pairs)})"
        )

    def __str__(self) -> str:
        """Return a human-readable summary of the document."""
        words_count = len(self.get_all_words())
        lines_count = len(self.get_all_lines())
        paragraphs_count = len(self.paragraphs)
        filename = os.path.basename(self.filepath) if self.filepath else "Unnamed document"
        mean_quality_score = sum(word.confidence for page in self.pages for word in page.words) / words_count if words_count > 0 else 0
        return (
            f"Document: {filename}\n"
            f"Status: {self.status}\n"
            f"Model: {self.model_id}\n"
            f"Created: {self.created_date_time}\n"
            f"Content: {len(self.pages)} pages, {words_count} words, {lines_count} lines\n"
            f"Elements: {len(self.tables)} tables, {len(self.figures)} figures, "
            f"{paragraphs_count} paragraphs, {len(self.key_value_pairs)} key-value pairs\n"
            f"Mean Quality Score: {mean_quality_score:.2f}"
        )

    def load_from_file(self, filepath: str) -> None:
        """Load OCR data from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.load_from_dict(data)
        except Exception as e:
            logger.error(f"Error loading OCR data from {filepath}: {e}")
            raise
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load OCR data from a dictionary."""
        try:
            self.raw_data = data
            self.status = data.get('status', '')
            self.created_date_time = data.get('createdDateTime', '')
            self.last_updated_date_time = data.get('lastUpdatedDateTime', '')
            
            analyze_result = data.get('analyzeResult', {})
            self.api_version = analyze_result.get('apiVersion', '')
            self.model_id = analyze_result.get('modelId', '')
            self.content = analyze_result.get('content', '')
            self.content_format = analyze_result.get('contentFormat', '')
            self.string_index_type = analyze_result.get('stringIndexType', '')
            
            # Parse pages
            if 'pages' in analyze_result:
                for page_data in analyze_result['pages']:
                    self._parse_page(page_data)
            
            # Parse tables
            if 'tables' in analyze_result:
                self._parse_tables(analyze_result['tables'])
            
            # Parse key-value pairs
            if 'keyValuePairs' in analyze_result:
                self._parse_key_value_pairs(analyze_result['keyValuePairs'])
            
            # Parse paragraphs
            if 'paragraphs' in analyze_result:
                self._parse_paragraphs(analyze_result['paragraphs'])
            
            # Parse figures
            if 'figures' in analyze_result:
                self._parse_figures(analyze_result['figures'])
            
            # Parse sections
            if 'sections' in analyze_result:
                self._parse_sections(analyze_result['sections'])
            
            # Parse styles
            if 'styles' in analyze_result:
                self._parse_styles(analyze_result['styles'])
            
            # Parse languages
            if 'languages' in analyze_result:
                self._parse_languages(analyze_result['languages'])
            
            logger.info(f"Successfully loaded OCR data with {len(self.pages)} pages")
        except Exception as e:
            logger.error(f"Error parsing OCR data: {e}")
            raise
    
    def _parse_page(self, page_data: Dict[str, Any]) -> None:
        """Parse a page from OCR data."""
        # Create spans
        spans = []
        if 'spans' in page_data:
            for span_data in page_data['spans']:
                spans.append(Span(
                    offset=span_data.get('offset', 0),
                    length=span_data.get('length', 0)
                ))
        
        # Parse words
        words = []
        if 'words' in page_data:
            for word_data in page_data['words']:
                span_data = word_data.get('span', {})
                span = Span(
                    offset=span_data.get('offset', 0),
                    length=span_data.get('length', 0)
                )
                
                word = OCRWord(
                    content=word_data.get('content', ''),
                    confidence=word_data.get('confidence', 0.0),
                    polygon=word_data.get('polygon', []),
                    span=span
                )
                words.append(word)
        
        # Parse lines
        lines = []
        if 'lines' in page_data:
            for line_data in page_data['lines']:
                line_spans = []
                for span_data in line_data.get('spans', []):
                    line_spans.append(Span(
                        offset=span_data.get('offset', 0),
                        length=span_data.get('length', 0)
                    ))
                
                line = OCRLine(
                    content=line_data.get('content', ''),
                    polygon=line_data.get('polygon', []),
                    spans=line_spans
                )
                
                # Add words to line based on span positions
                for word in words:
                    word_end = word.span.offset + word.span.length
                    for span in line_spans:
                        span_end = span.offset + span.length
                        if (word.span.offset >= span.offset and word_end <= span_end):
                            line.words.append(word)
                            break
                
                lines.append(line)
        
        # Create page
        page = OCRPage(
            page_number=page_data.get('pageNumber', 0),
            angle=page_data.get('angle', 0.0),
            width=page_data.get('width', 0.0),
            height=page_data.get('height', 0.0),
            unit=page_data.get('unit', 'inch'),
            words=words,
            lines=lines,
            spans=spans
        )
        
        self.pages.append(page)
    
    def _parse_tables(self, tables_data: List[Dict[str, Any]]) -> None:
        """Parse tables from OCR data."""
        for table_data in tables_data:
            # Parse bounding regions
            bounding_regions = []
            for region_data in table_data.get('boundingRegions', []):
                bounding_regions.append(BoundingRegion(
                    page_number=region_data.get('pageNumber', 0),
                    polygon=region_data.get('polygon', [])
                ))
            
            # Parse spans
            spans = []
            for span_data in table_data.get('spans', []):
                spans.append(Span(
                    offset=span_data.get('offset', 0),
                    length=span_data.get('length', 0)
                ))
            
            # Parse cells
            cells = []
            for cell_data in table_data.get('cells', []):
                cell_regions = []
                for region_data in cell_data.get('boundingRegions', []):
                    cell_regions.append(BoundingRegion(
                        page_number=region_data.get('pageNumber', 0),
                        polygon=region_data.get('polygon', [])
                    ))
                
                cell_spans = []
                for span_data in cell_data.get('spans', []):
                    cell_spans.append(Span(
                        offset=span_data.get('offset', 0),
                        length=span_data.get('length', 0)
                    ))
                
                cell = TableCell(
                    content=cell_data.get('content', ''),
                    row_index=cell_data.get('rowIndex', 0),
                    column_index=cell_data.get('columnIndex', 0),
                    kind=cell_data.get('kind', ''),
                    bounding_regions=cell_regions,
                    spans=cell_spans
                )
                cells.append(cell)
            
            table = OCRTable(
                row_count=table_data.get('rowCount', 0),
                column_count=table_data.get('columnCount', 0),
                cells=cells,
                bounding_regions=bounding_regions,
                spans=spans
            )
            
            self.tables.append(table)
    
    def _parse_key_value_pairs(self, kvp_data: List[Dict[str, Any]]) -> None:
        """Parse key-value pairs from OCR data."""
        for kvp in kvp_data:
            self.key_value_pairs.append(KeyValuePair(
                confidence=kvp.get('confidence', 0.0),
                key=kvp.get('key', {}),
                value=kvp.get('value', {})
            ))
    
    def _parse_paragraphs(self, paragraphs_data: List[Dict[str, Any]]) -> None:
        """Parse paragraphs from OCR data."""
        for paragraph_data in paragraphs_data:
            # Parse bounding regions
            bounding_regions = []
            for region_data in paragraph_data.get('boundingRegions', []):
                bounding_regions.append(BoundingRegion(
                    page_number=region_data.get('pageNumber', 0),
                    polygon=region_data.get('polygon', [])
                ))
            
            # Parse spans
            spans = []
            for span_data in paragraph_data.get('spans', []):
                spans.append(Span(
                    offset=span_data.get('offset', 0),
                    length=span_data.get('length', 0)
                ))
            
            paragraph = Paragraph(
                content=paragraph_data.get('content', ''),
                bounding_regions=bounding_regions,
                spans=spans
            )
            
            self.paragraphs.append(paragraph)
    
    def _parse_figures(self, figures_data: List[Dict[str, Any]]) -> None:
        """Parse figures from OCR data."""
        for figure_data in figures_data:
            # Parse bounding regions
            bounding_regions = []
            for region_data in figure_data.get('boundingRegions', []):
                bounding_regions.append(BoundingRegion(
                    page_number=region_data.get('pageNumber', 0),
                    polygon=region_data.get('polygon', [])
                ))
            
            # Parse spans
            spans = []
            for span_data in figure_data.get('spans', []):
                spans.append(Span(
                    offset=span_data.get('offset', 0),
                    length=span_data.get('length', 0)
                ))
            
            figure = Figure(
                id=figure_data.get('id', ''),
                bounding_regions=bounding_regions,
                spans=spans,
                elements=figure_data.get('elements', [])
            )
            
            self.figures.append(figure)
    
    def _parse_sections(self, sections_data: List[Dict[str, Any]]) -> None:
        """Parse sections from OCR data."""
        for section_data in sections_data:
            # Parse spans
            spans = []
            for span_data in section_data.get('spans', []):
                spans.append(Span(
                    offset=span_data.get('offset', 0),
                    length=span_data.get('length', 0)
                ))
            
            section = Section(
                spans=spans,
                elements=section_data.get('elements', [])
            )
            
            self.sections.append(section)
    
    def _parse_styles(self, styles_data: List[Dict[str, Any]]) -> None:
        """Parse styles from OCR data."""
        for style_data in styles_data:
            # Parse spans
            spans = []
            for span_data in style_data.get('spans', []):
                spans.append(Span(
                    offset=span_data.get('offset', 0),
                    length=span_data.get('length', 0)
                ))
            
            style = Style(
                confidence=style_data.get('confidence', 0.0),
                is_handwritten=style_data.get('isHandwritten', False),
                spans=spans
            )
            
            self.styles.append(style)
    
    def _parse_languages(self, languages_data: List[Dict[str, Any]]) -> None:
        """Parse languages from OCR data."""
        for language_data in languages_data:
            # Parse spans
            spans = []
            for span_data in language_data.get('spans', []):
                spans.append(Span(
                    offset=span_data.get('offset', 0),
                    length=span_data.get('length', 0)
                ))
            
            language = Language(
                locale=language_data.get('locale', ''),
                confidence=language_data.get('confidence', 0.0),
                spans=spans
            )
            
            self.languages.append(language)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the OCR document back to a dictionary."""
        # This is a placeholder that returns the original data
        # In a full implementation, you would serialize all the structured data back to JSON
        return dict(self.raw_data)
    
    def save_to_file(self, output_path: str) -> None:
        """Save the OCR document to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Successfully saved OCR data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving OCR data to {output_path}: {e}")
            raise
    
    def get_all_words(self) -> List[OCRWord]:
        """Get a flat list of all words across all pages."""
        return [word for page in self.pages for word in page.words]
    
    def get_all_lines(self) -> List[OCRLine]:
        """Get a flat list of all lines across all pages."""
        return [line for page in self.pages for line in page.lines]
    
    def get_page(self, page_number: int) -> Optional[OCRPage]:
        """Get a specific page by its number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None
    
    def get_key_value_pairs_by_key(self, key_text: str) -> List[KeyValuePair]:
        """Get all key-value pairs with a specific key text."""
        return [kvp for kvp in self.key_value_pairs 
                if kvp.key_text.lower() == key_text.lower()]
    
    def get_all_tables(self) -> List[OCRTable]:
        """Get all tables from the document."""
        return self.tables
    
    def get_all_figures(self) -> List[Figure]:
        """Get all figures from the document."""
        return self.figures
    
    def get_all_paragraphs(self) -> List[Paragraph]:
        """Get all paragraphs from the document."""
        return self.paragraphs


def parse_ocr_json(filepath: str) -> JSONParser:
    """Parse an OCR JSON file and return an OCRDocument object.
    
    Args:
        filepath: Path to the OCR JSON file
        
    Returns:
        An OCRDocument object containing the parsed data
    """
    return JSONParser(filepath)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        doc = parse_ocr_json(file_path)
        print(f"Loaded document with {len(doc.pages)} pages")
        
        # Print some basic stats
        total_words = len(doc.get_all_words())
        total_lines = len(doc.get_all_lines())
        total_tables = len(doc.get_all_tables())
        total_kvps = len(doc.key_value_pairs)
        total_figures = len(doc.get_all_figures())
        total_paragraphs = len(doc.get_all_paragraphs())
        
        print(f"Total words: {total_words}")
        print(f"Total lines: {total_lines}")
        print(f"Total tables: {total_tables}")
        print(f"Total key-value pairs: {total_kvps}")
        print(f"Total figures: {total_figures}")
        print(f"Total paragraphs: {total_paragraphs}")
        
        # Example of accessing a key-value pair
        if doc.key_value_pairs:
            first_kvp = doc.key_value_pairs[0]
            print(f"First key-value pair: '{first_kvp.key_text}' -> '{first_kvp.value_text}'")
            
        # Example of accessing a table
        if doc.tables:
            first_table = doc.tables[0]
            print(f"First table: {first_table.row_count} rows x {first_table.column_count} columns")
    else:
        print("Please provide a path to an OCR JSON file")