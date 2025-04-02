import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # or hardcode it if needed
FONTS_DIR = os.path.join(PROJECT_ROOT, 'fonts')

def _load_json(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'r') as f:
        return json.load(f)

def _resolve_typeface_paths(font_dict):
    return {
        name: os.path.join(FONTS_DIR, filename)
        for name, filename in font_dict.items()
    }

# Load and resolve
_typeface_config = _load_json('typefaces.json')
TYPEFACE_PATHS = _resolve_typeface_paths(_typeface_config)

# specify the variables to load into the namespace
__all__ = ['TYPEFACE_PATHS']
