from flask import Flask, send_from_directory, request, jsonify

from ml.ml import predict_ml

app = Flask(
    __name__,
    static_url_path='/static',
    static_folder='static/static',
)


@app.route('/', methods=['GET'])
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/ml/predict', methods=['POST'])
def predict():
    image = request.json['image']
    width = request.json['width']
    height = request.json['height']

    if width > 300 or height > 300:
        return jsonify({"result": "Too big image", "state": "400"}), 400

    result, prediction = predict_ml(image, width, height)
    return jsonify({"result": result, "pred": prediction, "state": "200"}), 200


@app.errorhandler(404)
def not_found(e):
    return '404', 404


if __name__ == '__main__':
    app.run(port=5123)
