from flask import Flask, render_template, request, url_for
from keras.applications import ResNet50
from keras_preprocessing.image import img_to_array
from keras_applications import imagenet_utils
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf

#template_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#print("template_dir: ", template_dir)
app = Flask(__name__)  #, template_folder=template_dir)
UPLOAD_FOLDER = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



model = None

def load_model():
    """
    load pretrained model - ResNet50
    """
    global model
    model = ResNet50(weights="imagenet");
    global graph
    graph = tf.get_default_graph()

def prepare_image(image, target):
    # image mode should be "RGB"
    if image.mode != "RGB":
        image = image.convert("RGB");

    # resize for model 
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return it
    return image

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(f)


    # return render_template('index.html')
    f = os.path.join("\\", f)
    print("uploaded_image", f)
    show_image = url_for('static', filename = file.filename)
    return render_template('result.html', uploaded_image=show_image)
    

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False};

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image, target=(224,224))

            with graph.as_default():

                preds = model.predict(image)
                results = imagenet_utils.decode_predictions(preds)
                data["predictions"] = []
                
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

                data["success"] = True
    
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask staring server..."
        "peases wait until server has fully started"))
    load_model()
    app.run()

