import os
import tempfile
import shutil
import pdfplumber  # Library for working with PDF files
import logging
import streamlit as st
from typing import List, Any

# Configure logging for the application, specifying log level, format, and date format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Initialize a logger for this module
logger = logging.getLogger(__name__)

@st.cache_data  # Cache the extracted images to improve performance on repeated runs
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    # Log the name of the uploaded file being processed
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")

    # Initialize a list to hold images of each page in the PDF
    pdf_pages = []
    
    # Open the uploaded PDF file with pdfplumber
    with pdfplumber.open(file_upload) as pdf:
        # Convert each page of the PDF into an image and store in the list
        pdf_pages = [page.to_image().original for page in pdf.pages]
    
    # Log successful extraction of pages as images
    logger.info("PDF pages extracted as images")
    
    return pdf_pages
