from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
from datetime import datetime

app = Flask(__name__)
model = load_model('Fruits_Classification.h5')  ## Ganti nama model file dengan RPS
target_img = os.path.join(os.getcwd() , 'static/images')

@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):

    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)

            # Mengukur waktu awal prediksi
            start_time = datetime.now()

            img = read_image(file_path)
            class_prediction = model.predict(img)
            classes_x = np.argmax(class_prediction, axis=1)

            # Mengukur waktu akhir prediksi
            end_time = datetime.now()

            # Menghitung lama waktu prediksi
            elapsed_time = end_time - start_time
            prediction_time = elapsed_time.total_seconds() * 1000  # Dalam milidetik

            if classes_x == 0:
                fruit = "Apple"
            elif classes_x == 1:
                fruit = "Banana"
            else:
                fruit = "Orange"

            # Menghitung akurasi prediksi dalam bentuk persen
            accuracy_percent = class_prediction[0][classes_x] * 100

            # Menambahkan akurasi prediksi (persen) dan lama waktu prediksi ke dalam template
            return render_template('predict.html', fruit=fruit, prob=class_prediction,
                                   accuracy_percent=accuracy_percent, prediction_time=prediction_time,
                                   user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"

        
        

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)