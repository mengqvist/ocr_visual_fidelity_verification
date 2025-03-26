# Data Directory

Storage for all data files used by the OCR Verification Tool.

## Structure

- **raw_json/**: Original OCR JSON files from Azure Document Intelligence
- **processed_json/**: Verification results and improved OCR data
- **pdfs/**: Original scanned PDF documents


## Azure Document Intelligence JSON Structure

The OCR JSON files from Azure Document Intelligence follow this hierarchical structure:

### analyzeResult
- apiVersion
- content
- contentFormat
- modelId
- stringIndexType

### pages
- pageNumber
- angle
- height
- width
- unit
- words
  - content
  - confidence
  - polygon
  - span
    - offset
    - length
- lines
  - content
  - polygon
  - spans
    - offset
    - length
- spans
  - offset
  - length

### tables
- rowCount
- columnCount
- boundingRegions
  - pageNumber
  - polygon
- cells
  - rowIndex
  - columnIndex
  - content
  - kind
  - boundingRegions
    - pageNumber
    - polygon
  - spans
    - offset
    - length
- spans
  - offset
  - length

### paragraphs
- content
- boundingRegions
  - pageNumber
  - polygon
- spans
  - offset
  - length

### keyValuePairs
- confidence
- key
  - content
  - boundingRegions
    - pageNumber
    - polygon
  - spans
    - offset
    - length
- value
  - content
  - boundingRegions
    - pageNumber
    - polygon
  - spans
    - offset
    - length

### figures
- id
- boundingRegions
  - pageNumber
  - polygon
- elements
- spans
  - offset
  - length

### sections
- elements
- spans
  - offset
  - length

### styles
- confidence
- isHandwritten
- spans
  - offset
  - length

### languages
- locale
- confidence
- spans
  - offset
  - length