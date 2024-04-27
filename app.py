from flask import Flask, request, jsonify, render_template
from services.data_preparation.data_cleaning import clean_dataset
from services.backend_setup.backend_setup import create_vector_database, evaluate_similarity_metrics
from data_preparation.web_scraping import scrape_product_images
from services.data_preparation import prepare_training_data
from services.cnn_model import build_cnn_model
import numpy as np


app = Flask(__name__)

# Paths to training data CSV and image directory
CSV_FILE = 'data/CNN_Model_Train_Data.csv'
IMG_SIZE = (224, 224)  # Example size, adjust as needed

# Prepare training data
X_train, X_val, y_train, y_val = prepare_training_data(CSV_FILE, img_size=IMG_SIZE)

# Build CNN model
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)  # Assuming 3 color channels (RGB)
num_classes = len(np.unique(y_train))  # Number of unique labels
model = build_cnn_model(input_shape, num_classes)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


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

# Define the route for the image-based product detection endpoint
@app.route('/image-product-detection', methods=['POST'])
def image_product_detection():
    # Get product image from request
    product_image = request.files.get('product_image')

    # Load the trained CNN model
    model = load_cnn_model()  # Function to load the trained CNN model

    # Preprocess the product image
    image = load_img(product_image, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values

    # Use the trained CNN model to predict product label
    predicted_label = model.predict(image)
    predicted_label = np.argmax(predicted_label)  # Get index of the predicted label

    # Match the predicted label with product description using the vector database
    # Replace the following line with your implementation for matching products
    matching_products = ["Product A", "Product B", "Product C"]  # Example matching products

    # Prepare the response
    response = {
        "predicted_product_description": "Product Description",  # Replace with actual product description
        "matching_products": matching_products
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
