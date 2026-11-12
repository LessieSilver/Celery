import pytest
from app import app, celery


@pytest.fixture
def client():
    app.config['TESTING'] = True
    celery.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        task_store_eager_result=True
    )
    with app.test_client() as client:
        yield client


def test_index(client):
    resp = client.get('/')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'endpoints' in data


def test_upscale_missing_field(client):
    resp = client.post('/upscale')
    assert resp.status_code == 400
    assert "No 'image' field" in resp.get_json()['error']


def test_upscale_and_get_result(client):
    import cv2
    import numpy as np
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.png', img)
    image_bytes = buf.tobytes()

    from io import BytesIO
    data = {'image': (BytesIO(image_bytes), 'test.png')}

    resp = client.post('/upscale', data=data, content_type='multipart/form-data')
    assert resp.status_code == 202
    task_id = resp.get_json()['task_id']

    resp = client.get(f'/tasks/{task_id}')
    assert resp.status_code == 200
    status = resp.get_json()['status']
    assert status == 'SUCCESS'

    resp = client.get(f'/processed/{task_id}')
    assert resp.status_code == 200
    assert resp.mimetype == 'image/png'
    assert len(resp.data) > 0