import json
import xml.etree.ElementTree as ET

import httpx
import pytest

from autolabel.detectors import Detection
from autolabel.label_studio import LabelStudio, build_tasks, labeling_config, local_file_url
from autolabel.labeling import ImageLabels


def test_labeling_config_is_valid_xml_with_one_label_per_class():
    names = ["verde", 'con "comillas"', "a&b", "<raro>"]
    root = ET.fromstring(labeling_config(names))            # falla si el XML está mal escapado
    assert [lab.get("value") for lab in root.iter("Label")] == names
    assert root.find("Image").get("value") == "$image"


def test_task_boxes_are_percentages_of_the_image():
    item = ImageLabels("a.png", 200, 100, (Detection("x", (20, 10, 60, 50), 0.8),))
    task = build_tasks([item], lambda n: f"/u/{n}", "m@0.2")[0]
    assert task["data"]["image"] == "/u/a.png"
    result = task["predictions"][0]["result"][0]
    assert result["type"] == "rectanglelabels"
    assert (result["original_width"], result["original_height"]) == (200, 100)
    assert result["value"]["x"] == pytest.approx(10)
    assert result["value"]["y"] == pytest.approx(10)
    assert result["value"]["width"] == pytest.approx(20)
    assert result["value"]["height"] == pytest.approx(40)
    assert result["value"]["rectanglelabels"] == ["x"]
    assert task["predictions"][0]["model_version"] == "m@0.2"
    assert task["predictions"][0]["score"] == pytest.approx(0.8)


def test_extra_fields_become_task_columns_without_touching_the_image():
    item = ImageLabels("a.png", 100, 100, ())
    extras = {"a.png": {"uncertain_boxes": 3, "confirmed_boxes": 5, "reason": "cajas dudosas: 3"}}
    task = build_tasks([item], lambda n: f"/u/{n}", "m", extras.get)[0]
    assert task["data"] == {"image": "/u/a.png", "uncertain_boxes": 3, "confirmed_boxes": 5,
                            "reason": "cajas dudosas: 3"}
    plain = build_tasks([item], lambda n: f"/u/{n}", "m", {}.get)[0]       # imagen sin datos extra
    assert plain["data"] == {"image": "/u/a.png"}


def test_image_without_detections_has_a_task_but_no_prediction():
    task = build_tasks([ImageLabels("a.png", 10, 10, ())], lambda n: n, "m")[0]
    assert "predictions" not in task


def test_local_file_url():
    url = local_file_url("/data/dataset/images/a.png", "/data")
    assert url == "/data/local-files/?d=dataset/images/a.png"


def test_login_posts_the_form_with_the_csrf_token_and_sets_the_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, headers={"set-cookie": "csrftoken=abc123; Path=/"})
        seen["form"] = request.content.decode()
        return httpx.Response(302, headers={"location": "/", "set-cookie": "sessionid=s1; Path=/"})

    client = LabelStudio("http://ls", transport=httpx.MockTransport(handler))
    client.login("a@b.c", "pw")
    assert "email=a%40b.c" in seen["form"] and "csrfmiddlewaretoken=abc123" in seen["form"]
    assert client._http.headers["X-CSRFToken"] == "abc123"


def test_login_failure_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<form>")        # vuelve el formulario: credenciales malas

    with pytest.raises(PermissionError):
        LabelStudio("http://ls", transport=httpx.MockTransport(handler)).login("x", "y")


def fake_server(state):
    """Simula la parte de la API de Label Studio que usa el cliente."""

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"].append((request.method, request.url.path))
        assert request.headers["Authorization"] == "Token t0k"
        path, method = request.url.path, request.method
        if path == "/health":
            return httpx.Response(200, json={"status": "UP"})
        if path == "/api/projects" and method == "GET":
            return httpx.Response(200, json={"results": state["projects"]})
        if path == "/api/projects" and method == "POST":
            state["projects"].append({"id": 7, "title": json.loads(request.content)["title"]})
            return httpx.Response(201, json={"id": 7})
        if path.startswith("/api/projects/") and method == "PATCH":
            state["config"] = json.loads(request.content)["label_config"]
            return httpx.Response(200, json={})
        if path == "/api/storages/localfiles" and method == "GET":
            return httpx.Response(200, json=state["storages"])
        if path == "/api/storages/localfiles" and method == "POST":
            state["storages"].append(json.loads(request.content))
            return httpx.Response(201, json={})
        if path.endswith("/import"):
            return httpx.Response(201, json={"task_count": len(json.loads(request.content))})
        if path.endswith("/tasks") and method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_client_creates_the_project_and_storage_once():
    state = {"calls": [], "projects": [], "storages": []}
    client = LabelStudio("http://ls", "t0k", transport=fake_server(state))
    client.wait_ready(seconds=1)

    pid = client.ensure_project("demo", "<View/>")
    assert pid == 7
    client.ensure_local_storage(pid, "/data/imgs")
    client.ensure_local_storage(pid, "/data/imgs")                # idempotente
    assert len(state["storages"]) == 1

    assert client.ensure_project("demo", "<View>nuevo</View>") == 7   # existe: actualiza la plantilla
    assert state["config"] == "<View>nuevo</View>"
    assert len(state["projects"]) == 1

    assert client.import_tasks(pid, [{"data": {}}, {"data": {}}]) == 2
    client.delete_tasks(pid)
