from setuptools import setup, find_packages
import os

# Read the long description from README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='vfv',
    version='0.1.0',
    description='Post-verification of OCR outputs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Martin Engqvist',
    author_email='martin.engqvist@gmail.com',
    packages=find_packages(include=['vfv', 'vfv.*']),
    include_package_data=True,  # Ensures non-code files specified in MANIFEST.in are included
    install_requires=[
        # List your runtime dependencies here, e.g.:
        # 'numpy>=1.19.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
