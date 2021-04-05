from flask import Flask, render_template, request
from modelhandling import get_prediction

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    imagefile = request.files["imagefile"]
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    preds = get_prediction(image_path)
    return render_template("index.html", predictions=preds)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
