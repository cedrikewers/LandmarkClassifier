from flask import Flask, request, render_template, send_file
from werkzeug.middleware.proxy_fix import ProxyFix
import torch
import pandas as pd
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, EfficientNet, ResNet, resnet50, ResNet50_Weights, MobileNet_V3_Small_Weights, MobileNetV3, mobilenet_v3_small
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable, Literal, Any
import mistune
from urllib.parse import unquote
from manual_mapping import manual_mapping
import os
import io
import base64
from mpl_toolkits.basemap import Basemap
import json
import matplotlib.pyplot as plt
import matplotlib
from werkzeug.datastructures import FileStorage
import os

markdown = mistune.create_markdown(plugins=['table'])

base_url_path = ""

if "BASE_URL_PATH" in os.environ and not base_url_path:
    base_url_path = os.environ["BASE_URL_PATH"]

type TransformFunction = Callable[[Image.Image], torch.Tensor]

N_CLASSES = 10_000

name_mapping = pd.read_csv("train_label_to_hierarchical.csv")

with open("geolocations.json") as f:
    geolocations = json.loads(f.read())

model_file_from_name = {
    # "efficientnet-b0-v2": "b0-10_000v2.pth",
    "efficientnet-b0-v3": "b0-10_000v3.pth",
    "efficientnet-b0-v4": "b0-10_000v4.pth",
    # "ResNet50-v1": "ResNet50-10_000v1.pth",
    "ResNet50-v2": "ResNet50-10_000v2.pth",
    # "MobileNet_V3_Small-v1": "MobileNet_V3_Small-10_000v1.pth",
    "MobileNet_V3_Small-v2": "MobileNet_V3_Small-10_000v2.pth",
}

def load_model(file_name: str) -> tuple[EfficientNet | ResNet | MobileNetV3, TransformFunction]:
    
    model_type: Literal["efficientnet-b0", "ResNet50", "MobileNet_V3_Small"] = "efficientnet-b0"
    if "ResNet50" in file_name:
        model_type = "ResNet50"
    elif "MobileNet_V3_Small" in file_name:
        model_type = "MobileNet_V3_Small"
    
    
    file_name = f"models/{file_name}"
    if model_type == "efficientnet-b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, N_CLASSES)
        model.load_state_dict(torch.load(file_name, map_location=torch.device("cpu")))
    elif model_type == "ResNet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50()
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
        model.load_state_dict(torch.load(file_name, map_location=torch.device("cpu")))
    elif model_type == "MobileNet_V3_Small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, N_CLASSES)
        model.load_state_dict(torch.load(file_name, map_location=torch.device("cpu")))
    else:
        raise ValueError("Invalid model type")
        
    model.eval()
    return model, weights.transforms()

with open("mapping.dict") as f:
    text = f.read()
mapping = eval(text)
mapping = {int(v): k for k, v in mapping.items()}

def get_landmark_name(landmark_id: int) -> str | None:
    try:
        (predicted_url,) = name_mapping.where(
            name_mapping["landmark_id"] == landmark_id
        ).dropna()["category"]
        name = predicted_url.split(":")[2]
        name = unquote(name)
    except Exception:
        if landmark_id in manual_mapping.keys():
            name = manual_mapping[landmark_id]
        else:
            name = None
    return name


app = Flask(__name__)

# Configure ProxyFix to handle reverse proxy headers correctly
# This fixes the issue where Flask generates localhost URLs when behind a reverse proxy
app.wsgi_app = ProxyFix(
    app.wsgi_app, 
    x_for=1,     # Trust 1 proxy for X-Forwarded-For
    x_proto=1,   # Trust 1 proxy for X-Forwarded-Proto (http/https)
    x_host=1,    # Trust 1 proxy for X-Forwarded-Host
    x_prefix=1   # Trust 1 proxy for X-Forwarded-Prefix
)


@app.route(f"{base_url_path}/")
def index():
    return render_template("index.jinja", models=model_file_from_name.keys(), base_url_path=base_url_path)
    
@dataclass
class PredictionResult:
    landmark_ids: list[int] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    landmark_names: list[str | None] = field(default_factory=list)
    
    def __repr__(self) -> str:
        out = "| Landmark ID | Prob | Landmark Name |\n|------------:|----:|-------------:|\n"
        for landmark_id, prob, landmark_name in zip(self.landmark_ids, self.probabilities, self.landmark_names):
            landmark_id = f"[{landmark_id}]({base_url_path}/label_sample/{landmark_id}/)"
            out += f"|{landmark_id: >5}|{prob: >5.2%}|{landmark_name}|\n"
        return out
    
    
def classify(img: FileStorage, model: ResNet | EfficientNet | MobileNetV3, transform: TransformFunction) -> PredictionResult:
    with torch.no_grad():
        image = Image.open(img).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)
        output = model(image)
        
        probabilities = F.softmax(output, dim=1)
        top_probabilities, top_indices = torch.topk(probabilities, 5, dim=1)
        top_probabilities, top_indices = top_probabilities.tolist(), top_indices.tolist()
        
        res = PredictionResult()
        for prop, index in zip(top_probabilities[0], top_indices[0]):
            res.landmark_ids.append(mapping[index])
            res.probabilities.append(prop)
            res.landmark_names.append(get_landmark_name(mapping[index]))
    return res

def get_bae64_image(img: Any) -> str:
    image = Image.open(img)
    
    aspect_ratio = image.width / image.height
    height = 500
    width = int(aspect_ratio * height)
    image = image.resize((width, height))
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return f'<br style="margin-bottom:10px"><img src="data:image/png;base64,{img_base64}"/>'

def get_geolocation_image(pr: PredictionResult) -> str:
    
    colors = ["gold", "yellowgreen","purple",  "royalblue", "crimson"]
    
    matplotlib.use('agg')
    fig = plt.figure(figsize=(8, 5))
    m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    xy_coords = {key: m(val["lng"], val["lat"]) for key, val in geolocations.items()}
    for i, (landmark_id, landmark_name) in enumerate(zip(pr.landmark_ids[::-1], pr.landmark_names[::-1])):
        if str(landmark_id) in xy_coords.keys():
            x, y = xy_coords[str(landmark_id)]
            m.plot(x, y, "o", markersize=(4+i), label=landmark_name, c=colors[i], alpha=0.85)
    fig.legend(reverse=True)
    fig.tight_layout()
    buffered = io.BytesIO()
    fig.savefig(buffered, format="PNG", dpi=100)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return f'<br style="margin-bottom:10px"><img src="data:image/png;base64,{img_base64}"/>'

@app.route(f"{base_url_path}/classify", methods=["POST"])
def classify_rotue():
    img = request.files["image"]
    model_name = request.form["model"]
    model_file = model_file_from_name[model_name]
    
    model, transform = load_model(model_file)
    
    res = classify(img, model, transform)
    
    base64image = get_bae64_image(img)
    geolocations_image = get_geolocation_image(res)
    
    html = markdown(repr(res))
    if not isinstance(html, str):
        return "Error in rendering markdown", 500
    html += base64image
    html += geolocations_image
    return render_template("results.jinja", html=html)

@app.route(f"{base_url_path}/label_sample/<landmark_id>/")
def get_label_sample_img(landmark_id):
    try:
        return send_file(os.path.join("label_samples", f"{landmark_id}.jpg"))
    except Exception as e:
        return str(e), 404


if __name__ == "__main__":
    app.run(port=5025, host="0.0.0.0", debug=False)
