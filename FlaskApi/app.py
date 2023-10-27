import pickle
import numpy as np
from numpy.linalg import norm
import cv2
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

# Load saved embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Create a model by adding GlobalMaxPooling2D layer
model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Load and preprocess the query image
query_img = image.load_img('query_images/R.jpg', target_size=(224, 224))
query_img_array = image.img_to_array(query_img)
expanded_query_img_array = np.expand_dims(query_img_array, axis=0)
preprocessed_query_img = preprocess_input(expanded_query_img_array)

# Get the normalized feature vector for the query image
query_result = model.predict(preprocessed_query_img).flatten()
normalized_query_result = query_result / norm(query_result)

# Fit NearestNeighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
# Find nearest neighbors for the query image
distances, indices = neighbors.kneighbors([normalized_query_result])

# Display images of nearest neighbors
for file_index in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file_index])
    if temp_img is not None:
        cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
        cv2.waitKey(0)
    else:
        print(f"Failed to load image: {filenames[file_index]}")

E_filenames = [filename.replace('images\\', '') for filename in filenames]
# print("Filenames:", E_filenames)
# Close all OpenCV windows after displaying
cv2.destroyAllWindows()

# Save the trained model for later use
# with open('Apparel_Recommendation_System.pickle', 'wb') as f:
#     pickle.dump(model, f)

# If you plan to use JSON serialization for model's columns
# import json
# columns = {
#     'data_columns': [col.lower() for col in X.columns]
# }
# with open('columns.json', 'w') as f:
#     json.dump(columns, f)
