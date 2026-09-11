import torch
from insightface.app import FaceAnalysis
import zipfile

# Define providers dynamically based on CUDA availability
providers = ['CPUExecutionProvider']
if torch.cuda.is_available():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
model_root = './' # ----
zip_path = f"{model_root}models/antelopev2.zip" # ----
extract_folder = f"{model_root}models/" # ----

try:                                                                        # ----
    FaceAnalysis(name='antelopev2', root=model_root, providers=providers)   # ----
    # FaceAnalysis(name='buffalo_l', root=model_root, providers=providers)    # ----
except AssertionError:                                                      # ----
    print("Error during zip extraction. Rectification in following steps")  # ----
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder) # ----
    print(f"Extracted {zip_path} to {extract_folder}/") # ----

# This snippet downloads the pretrained weights for Arc2Face Application. Found this on hugging face.
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/config.json", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/diffusion_pytorch_model.safetensors", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/config.json", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/pytorch_model.bin", local_dir="./models")

hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arcface.onnx", local_dir="./models/antelopev2")