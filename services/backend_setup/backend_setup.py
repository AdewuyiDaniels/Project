from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

def create_vector_database():
    # Initialize Pinecone client
    pinecone = Pinecone()
    
    # Define database schema and settings
    schema = {
        'product_id': 'int',
        'vector': 'float32'
    }
    settings = {
        'dimension': 100  # Adjust dimension as per your vector size
    }
    
    # Create the vector database
    database_name = 'product_vectors'
    pinecone.create_index(database_name, schema, settings=settings)
    
    return database_name

def evaluate_similarity_metrics(vectors):
    # Evaluate different similarity metrics
    cosine_sim = cosine_similarity(vectors)
    euclidean_dist = euclidean_distances(vectors)
    manhattan_dist = manhattan_distances(vectors)
    
    
    # Calculate average similarity scores
    avg_cosine_sim = cosine_sim.mean()
    avg_euclidean_dist = euclidean_dist.mean()
    avg_manhattan_dist = manhattan_dist.mean()
    
    # Choose the best similarity metric
    max_score = max(avg_cosine_sim, avg_euclidean_dist, avg_manhattan_dist)
    if max_score == avg_cosine_sim:
        selected_metric = 'cosine_similarity'
    elif max_score == avg_euclidean_dist:
        selected_metric = 'euclidean_distance'
    else:
        selected_metric = 'manhattan_distance'
    
    return selected_metric
