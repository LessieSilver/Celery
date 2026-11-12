import os
import base64
from flask import Flask, request, jsonify, Response, url_for
from celery import Celery
from celery.signals import worker_init

from upscale import upscale_bytes, _get_scaler

app = Flask(__name__)

BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0')
RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

app.config.update(
    broker_url=BROKER_URL,
    result_backend=RESULT_BACKEND,
)

celery = Celery(
    app.import_name,
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)
celery.conf.update(app.config)


@worker_init.connect
def init_worker(**kwargs):
    print("[worker_init] Loading EDSR model...")
    _get_scaler()
    print("[worker_init] Model loaded.")


@celery.task(name='app.process_image')
def process_image_task(image_b64: str):
    image_bytes = base64.b64decode(image_b64)
    result_bytes = upscale_bytes(image_bytes)
    return base64.b64encode(result_bytes).decode('utf-8')


@app.route('/')
def index():
    return jsonify({
        'message': 'Image Upscaling Service',
        'endpoints': {
            'POST /upscale': 'Upload image (field: image)',
            'GET /tasks/<task_id>': 'Get task status',
            'GET /processed/<task_id>': 'Download processed image',
        }
    })


@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'image' not in request.files:
        return jsonify({'error': "No 'image' field in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({'error': 'Empty file'}), 400

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    task = process_image_task.delay(image_b64)
    return jsonify({'task_id': task.id}), 202


@app.route('/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = process_image_task.AsyncResult(task_id)
    response = {'task_id': task_id, 'status': task.state}

    if task.state == 'SUCCESS':
        response['result_url'] = url_for('get_processed_file', task_id=task_id, _external=True)
    elif task.state == 'FAILURE':
        response['error'] = str(task.info)

    return jsonify(response)


@app.route('/processed/<task_id>', methods=['GET'])
def get_processed_file(task_id):
    task = process_image_task.AsyncResult(task_id)

    if task.state != 'SUCCESS':
        return jsonify({'error': 'Task not ready', 'status': task.state}), 400

    result_b64 = task.result
    result_bytes = base64.b64decode(result_b64)

    return Response(result_bytes, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)