import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

app = Flask(__name__)

model = load_model("Saravjeet_IBMDR.h5")

# Create the "uploads" directory if it doesn't exist
uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        filepath = os.path.join(uploads_dir, f.filename)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x), axis=1)
        index = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferative DR', 'Severe DR']
        text = "The Classified image is: " + str(index[pred[0]])
        return text

if __name__ == '__main__':
    app.run()
