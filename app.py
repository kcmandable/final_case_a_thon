import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import base64
from PIL import Image
import io
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_tiny
from ultralytics import YOLO
import json
import requests
import os

# ------------------- File Setup -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "benthic_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Local file paths
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "convnext_tiny_best.pth")
SECOND_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "benthic_yolov8_best.pt")
CLASSES_PATH = os.path.join(ARTIFACTS_DIR, "classes.json")

# Remote URLs
MODEL_URL = "https://huggingface.co/kcmandable/convnext_tiny_benthic/resolve/main/convnext_tiny_best.pth"
SECOND_MODEL_URL = "https://huggingface.co/kcmandable/benthic_yolov8/resolve/main/benthic_yolov8_best.pt"
CLASSES_URL = "https://huggingface.co/kcmandable/convnext_tiny_benthic/resolve/main/classes.json"

# ------------------- Download Helper -------------------
def download_if_missing(url, file_path):
    """Download file from Hugging Face if missing."""
    if os.path.exists(file_path):
        print(f"✅ Found {file_path}")
        return
    print(f"⬇️ Downloading {os.path.basename(file_path)} ...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded and saved to {file_path}")
    except Exception as e:
        print(f"❌ Could not download {file_path} from {url}")
        print(f"Error: {e}")
        raise FileNotFoundError(f"Missing required file: {file_path}")

# ------------------- Ensure Artifacts Exist -------------------
for url, path in [
    (MODEL_URL, MODEL_PATH),
    (SECOND_MODEL_URL, SECOND_MODEL_PATH),
    (CLASSES_URL, CLASSES_PATH),
]:
    download_if_missing(url, path)

# ------------------- Load Models -------------------
print("🚀 Loading models...")
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = convnext_tiny(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
model.to(device).eval()

second_model = YOLO(SECOND_MODEL_PATH)
print("✅ Models loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------------------- Utility Functions -------------------
def classify_image(image_array):
    image_pil = Image.fromarray(image_array.astype('uint8'), mode="RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_index = outputs.argmax(dim=1).item()
        return CLASS_NAMES[pred_index]

def draw_box_on_image(image_array):
    image_pil = Image.fromarray(image_array.astype('uint8'), mode="RGB")
    results = second_model(image_pil)
    boxed_image = results[0].plot()
    buffered = io.BytesIO()
    Image.fromarray(boxed_image).save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# ------------------- Dash Setup -------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
image_store = []

# ------------------- UI Components -------------------
banner = html.Div([
    html.Img(src='/assets/wm.png', className='banner-image'),
    html.Div("Benthic Species Recognition", className='banner-title'),
    html.Img(src='/assets/wm.png', className='banner-image')
], className='banner')

description_box = html.Div([
    html.H4("Description"),
    html.P(
        "This AI-powered tool identifies benthic species from underwater images using ConvNeXt for classification "
        "and YOLOv8 for detection. It recognizes species like Scallops, Roundfish, and Crabs."
    )
], className='info-box')

instruction_box = html.Div([
    html.H4("Instructions"),
    html.P(
        "Upload an underwater image below to view classification and detection results. "
        "Visit the Gallery to see past uploads."
    )
], className='info-box')

box_container = html.Div(
    [description_box, instruction_box],
    style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'marginTop': '20px'}
)

# ------------------- Pages -------------------
def upload_page():
    return html.Div([
        banner,
        box_container,
        dbc.Container([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    html.Span('Drag and Drop or ', className='upload-text'),
                    html.A('Select an Image', className='upload-text')
                ]),
                className='upload-box',
                accept='image/*',
                multiple=True
            ),
            dcc.Loading(
                id="loading-output",
                type="circle",
                color="#004080",
                children=html.Div(id='output-image-upload')
            ),
            html.Br(),
            html.Div(
                dbc.Button("Go to Gallery", href="/gallery", color="secondary", style={
                    'marginTop': '20px', 'marginBottom': '30px'
                }),
                style={'textAlign': 'center'}
            )
        ])
    ], className='center-box-wrapper')

def gallery_page():
    if not image_store:
        return html.Div([
            banner,
            dbc.Container([
                html.H3("Gallery of Uploaded Images", className='text-center', style={
                    'color': 'white', 'fontFamily': 'Times New Roman, serif',
                    'marginTop': '50px', 'fontSize': '30px'
                }),
                html.P("No images uploaded yet.", style={'textAlign': 'center', 'color': 'white'}),
                dbc.Button("Back to Upload", href="/", color="secondary", className='mt-3')
            ])
        ])

    gallery_items = [
        html.Div([
            html.H6(item.get('filename', 'Unknown'), style={'fontWeight': 'bold', 'color': '#003366'}),
            html.P(f"Prediction: {item['prediction']}"),
            html.P(f"Uploaded: {item['timestamp']}", style={'fontSize': '12px', 'color': '#666'}),
            html.Img(src=item['original']),
            html.Img(src=item['boxed'], style={'marginTop': '10px'})
        ], className='gallery-item') for item in image_store
    ]

    return html.Div([
        banner,
        dbc.Container([
            html.H3("Gallery of Uploaded Images", className='text-center', style={
                'color': 'white', 'fontFamily': 'Times New Roman, serif',
                'marginTop': '50px', 'fontSize': '30px'
            }),
            dbc.Button("Back to Upload", href="/", color="secondary", className='mb-4'),
            html.Div(gallery_items, style={
                'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'gap': '20px'
            })
        ])
    ])

# ------------------- Routing -------------------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/gallery':
        return gallery_page()
    return upload_page()

# ------------------- Upload Callback -------------------
@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
)
def update_output(contents, filenames):
    if contents is None:
        return
    if not isinstance(contents, list):
        contents, filenames = [contents], [filenames]
    children = []
    for content, filename in zip(contents, filenames):
        try:
            decoded = base64.b64decode(content.split(',')[1])
            image = Image.open(io.BytesIO(decoded)).convert("RGB")
            image_array = np.array(image)
            prediction = classify_image(image_array)
            boxed_image_b64 = draw_box_on_image(image_array)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            image_store.append({
                'original': content,
                'boxed': boxed_image_b64,
                'prediction': prediction,
                'timestamp': timestamp,
                'filename': filename
            })
            children.append(html.Div([
                html.H5(f'Prediction: {prediction}', style={'color': '#003366', 'fontWeight': 'bold'}),
                html.Div([
                    html.Img(src=content, style={'maxWidth': '220px', 'borderRadius': '8px'}),
                    html.Img(src=boxed_image_b64, style={'maxWidth': '220px', 'borderRadius': '8px', 'marginLeft': '20px'})
                ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px'}),
                html.P(f"Uploaded: {timestamp}", style={'fontSize': '13px', 'color': '#555'}),
                html.Hr()
            ], className='prediction-card'))
        except Exception as e:
            children.append(html.Div([html.P(f"Error processing {filename}: {str(e)}")]))
    return children

# ------------------- Run -------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
