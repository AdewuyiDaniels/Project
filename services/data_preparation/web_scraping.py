# data_preparation/web_scraping.py

import requests
from bs4 import BeautifulSoup
import os

def scrape_product_images(url, output_directory):
    """
    Scrape product images from an e-commerce website.
    
    Args:
    - url: URL of the e-commerce website.
    - output_directory: Directory to save the scraped images.
    
    Returns:
    - List of paths to the saved image files.
    """
    # Send GET request to the URL
    response = requests.get(url)
    
    # Parse HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find product image elements
    image_elements = soup.find_all('img', {'class': 'product-image'})
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Download and save images
    image_paths = []
    for idx, img in enumerate(image_elements):
        img_url = img['src']
        img_name = f"product_{idx}.jpg"
        img_path = os.path.join(output_directory, img_name)
        
        # Download image
        img_data = requests.get(img_url).content
        with open(img_path, 'wb') as f:
            f.write(img_data)
        
        image_paths.append(img_path)
    
    return image_paths
