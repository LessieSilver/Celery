import os
import uuid
from flask import Flask, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from celery import Celery
from upscale import upscale

app = Flask(__name__)

app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)


def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND']
    )
    celery.conf.update(app.config)
    return celery


celery = make_celery(app)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

MODEL_PATH = 'EDSR_x2.pb'


@celery.task(name='app.process_image')
def process_image_task(input_filepath, output_filepath):
    upscale(input_filepath, output_filepath, model_path=MODEL_PATH)

    return os.path.basename(output_filepath)



@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    input_filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(input_filepath)

    name, ext = os.path.splitext(unique_filename)
    output_filename = f"{name}_upscaled{ext}"
    output_filepath = os.path.join(PROCESSED_FOLDER, output_filename)

    task = process_image_task.delay(input_filepath, output_filepath)

    return jsonify({'task_id': task.id}), 202


@app.route('/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = process_image_task.AsyncResult(task_id)

    response = {
        'task_id': task_id,
        'status': task.state
    }

    if task.state == 'SUCCESS':
        output_filename = task.result
        result_url = url_for('get_processed_file', filename=output_filename, _external=True)
        response['result_url'] = result_url
    elif task.state == 'FAILURE':
        response['error'] = str(task.info)

    return jsonify(response)


@app.route('/processed/<path:filename>', methods=['GET'])
def get_processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)