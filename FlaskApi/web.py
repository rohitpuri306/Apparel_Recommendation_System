from flask import Flask, render_template, request, url_for
import os
import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import pickle
import cv2

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload and recommendation
        imagefile = request.files['imagefile']
        image_path = os.path.join('query_images', imagefile.filename)
        imagefile.save(image_path)

        # Load embeddings and filenames
        feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
        original_filenames = pickle.load(open('filenames.pkl', 'rb'))
        filenames = [filename.replace('images\\', '') for filename in original_filenames]

        # Load pre-trained ResNet50 model
        model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model.trainable = False
        model = tensorflow.keras.Sequential([
            model,
            GlobalMaxPooling2D()
        ])

        # Preprocess the uploaded image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        # Fit NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)

        distances, indices = neighbors.kneighbors([normalized_result])


        # Get the indices of recommended images
        recommended_indices = indices[0][1:]  # Exclude the first index, which is the query image itself

        # Generate image URLs for recommended images
        image_urls = [url_for('static', filename='images/' + filenames[index]) for index in recommended_indices]
        filenames = [filenames[index] for index in recommended_indices]

        return render_template('index.html', prediction=image_urls, filenames=filenames)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
