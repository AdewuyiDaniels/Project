from flask import Flask, request, jsonify, render_template
from services.data_preparation.data_cleaning import clean_dataset
from services.backend_setup.backend_setup import create_vector_database, evaluate_similarity_metrics
from data_preparation.web_scraping import scrape_product_images

app = Flask(__name__)

@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    """
    Endpoint for product recommendations based on natural language queries.
    Input: Form data containing 'query' (string).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    query = request.form.get('query', '')
    # Process the query to find matching products
    products = []  # Empty array, to be populated with product data
    response = ""  # Empty string, to be filled with a natural language response
    return jsonify({"products": products, "response": response})

@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    """
    Endpoint to process handwritten queries extracted from uploaded images.
    Input: Form data containing 'image_data' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    image_file = request.files.get('image_data')
    # Process the image to extract text and find matching products
    products = []  # Empty array, to be populated with product data
    response = ""  # Empty string, to be filled with a natural language response
    return jsonify({"products": products, "response": response})

@app.route('/image-product-search', methods=['POST'])
def image_product_search():
    """
    Endpoint to identify and suggest products from uploaded product images.
    Input: Form data containing 'product_image' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    product_image = request.files.get('product_image')
    # Process the product image to detect and match products
    products = []  # Empty array, to be populated with product data
    response = ""  # Empty string, to be filled with a natural language response
    return jsonify({"products": products, "response": response})

@app.route('/sample_response', methods=['GET'])
def sample_response():
    """
    Endpoint to return a sample JSON response for the API.
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    return render_template('sample_response.html')

@app.route('/clean-dataset', methods=['POST'])
def clean_dataset_endpoint():
    """
    Endpoint to clean the dataset.
    Input: Form data containing 'dataset_file' (file, zip or direct file upload).
    Output: JSON with 'cleaned_dataset_path' (string).
    """
    dataset_file = request.files.get('dataset_file')
    cleaned_dataset_path = clean_dataset(dataset_file)
    return jsonify({"cleaned_dataset_path": cleaned_dataset_path})

@app.route('/create-vector-db', methods=['POST'])
def create_vector_db_endpoint():
    """
    Endpoint to create a vector database.
    Output: JSON with 'database_name' (string).
    """
    database_name = create_vector_database()
    return jsonify({"database_name": database_name})

@app.route('/select-similarity-metric', methods=['POST'])
def select_similarity_metric_endpoint():
    """
    Endpoint to select a similarity metric.
    Output: JSON with 'selected_metric' (string).
    """
    # You may need to pass necessary data as input for selecting the similarity metric
    dataset_vectors = database_name  # Fetch dataset vectors
    selected_metric = select_similarity_metric(dataset_vectors)
    return jsonify({"selected_metric": selected_metric})

@app.route('/scrape-images', methods=['POST'])
def scrape_images():
    """
    Endpoint to trigger web scraping for product images.
    Input: JSON data containing 'url' and 'output_directory'.
    Output: JSON with 'image_paths'.
    """
    data = request.json
    url = data.get('url')
    output_directory = data.get('output_directory')

    if url and output_directory:
        image_paths = scrape_product_images(url, output_directory)
        return jsonify({"image_paths": image_paths}), 200
    else:
        return jsonify({"message": "URL and output directory are required."}), 400


if __name__ == '__main__':
    app.run(debug=True)
