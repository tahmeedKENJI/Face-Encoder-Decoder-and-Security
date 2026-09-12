from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

import torch
import os
import sys
import glob
import pathlib
from pathlib import Path
import random
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

sys.path.append("./Arc2Face")
from arc2face import CLIPTextModelWrapper, project_face_embs

#---------------------------------------------------------------------------------------------------
# ARC2FACE PIPELINE
#---------------------------------------------------------------------------------------------------

print("Building the Arc2Face Pipeline...")
base_model = 'runwayml/stable-diffusion-v1-5'

encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=torch.float16
)

unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=torch.float16
)

pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

print("Assigning Pipeline Scheduler...")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to('cuda')

#---------------------------------------------------------------------------------------------------
# FACE TEMPLATE GEN-APP
#---------------------------------------------------------------------------------------------------
print("Building Face Template Generator...")
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

app2 = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app2.prepare(ctx_id=0, det_size=(480, 480))

#---------------------------------------------------------------------------------------------------
# ARGUMENTS
#---------------------------------------------------------------------------------------------------
print("Parsing Command Line Arguments...")
image_path = str(sys.argv[1]) # folder path to images
startind = int(sys.argv[2]) if len(sys.argv) > 2 else 0

targetfiles = sorted(glob.glob(f'{image_path}/*.jpg'))
print(len(targetfiles))

id_emb_cumarray = []

#---------------------------------------------------------------------------------------------------
# GENERATE FACE TEMPLATES
#---------------------------------------------------------------------------------------------------
print("Generating Face Templates...")
for imgid, img in enumerate(targetfiles[startind:], start=startind):
    fname1= pathlib.PurePath(str(img))
    print(imgid, fname1)
    imgid_target = Path(str(fname1)).stem  

    try:
        img_target = np.array(Image.open(str(fname1)))[:,:,::-1]
        try:    
            faces_target = app.get(img_target)
            faces_target = sorted(faces_target, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
        except:
            faces_target = app2.get(img_target)
            faces_target = sorted(faces_target, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
        
        id_emb_target = torch.tensor(faces_target['embedding'], dtype=torch.float16)[None]
        # save outputs
        id_emb_nparr = id_emb_target.detach().cpu().numpy()
        id_emb_cumarray.append(id_emb_nparr)
    except:
        continue
      
 
templates = id_emb_cumarray 
print(len(templates))

#---------------------------------------------------------------------------------------------------
# DISRUPTOR GEN ROUTINES
#---------------------------------------------------------------------------------------------------
## positive disruptor generation
def add_noise(embedding, std=0.2):
    #noise = (var**0.5)*torch.randn(embedding.shape, dtype=torch.float16).cuda()
    #return embedding + noise
    return embedding + torch.randn_like(embedding) * std

def feature_dropout(embedding, p=0.8):
    mask = torch.bernoulli(torch.ones_like(embedding) * p)
    return embedding * mask

## negative feature/orthogonal disruptor generation
def gram_schmidt(v):
    """
    Finds an orthogonal vector to the given vector using the Gram-Schmidt process.
    """    
    u = torch.randn_like(v)  # Generate a random vector of the same size
    u = u - (torch.dot(u, v) / torch.dot(v, v)) * v
    #return u / torch.norm(u)
    return u

def synth_neg(folder_path):
   # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Choose a random file
    random_file = random.choice(files)

    # Construct the full path to the file
    file_path = os.path.join(folder_path, random_file)
    synth = torch.from_numpy(np.load(f"{file_path}", allow_pickle=True))
    synth = synth.to(torch.float16).cuda()
    return synth 

#---------------------------------------------------------------------------------------------------
# GEN ARGUMENTS
#---------------------------------------------------------------------------------------------------
folder_path = './StyleGAN3Templates' # Make sure to unzip StyleGAN3Templates, add the path here.
# save outputs
save_path = './dis_images'
os.makedirs(save_path, exist_ok=True)
num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 2

#---------------------------------------------------------------------------------------------------
# RECONSTRUCT FACES
#---------------------------------------------------------------------------------------------------
print("Reconstructing Unprotected Faces and Disruptors...")
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

for ind, tempfile in enumerate(templates):
    print(ind)
    id_emb_target = torch.from_numpy(tempfile).cuda()
    filename = f"sub_{ind}"
   
    ## reconstruct original template
    template = id_emb_target.to(torch.float16)
    template = id_emb_target/torch.norm(id_emb_target, dim=1, keepdim=True)   # normalize embedding 
    id_emb_template = project_face_embs(pipeline, template)    # pass through the encoder
    image = pipeline(prompt_embeds=id_emb_template, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=1).images
    image[0].save(f"{save_path}/{filename}_template_InsightFace.png")

    ## reconstruct positive disruptors

    # noise
    id_emb_noisy = add_noise(id_emb_target)# add noise to id_emb_target not template
    id_emb_noisy = id_emb_noisy/torch.norm(id_emb_noisy, dim=1, keepdim=True)   # normalize noisy embedding
    id_emb_noisy_project = project_face_embs(pipeline, id_emb_noisy)  
    print('cosine noisy:',cos(template, id_emb_noisy))
    images_noisy = pipeline(prompt_embeds=id_emb_noisy_project, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images
    for id, img in enumerate(images_noisy):
        img.save(f"{save_path}/{filename}_{id}_customnoise_InsightFace.png")
    
    # mask
    pos_mask = feature_dropout(id_emb_target)
    pos_mask = pos_mask/torch.norm(pos_mask, dim=1, keepdim=True)
    id_emb_mask_template = project_face_embs(pipeline, pos_mask)    # pass through the encoder
    images_mask = pipeline(prompt_embeds=id_emb_mask_template, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images
    print('cosine mask:',cos(template, pos_mask))
    for id, img in enumerate(images_mask):    
        img.save(f"{save_path}/{filename}_{id}_custommask_InsightFace.png")

    ## reconstruct negative disruptors
    # orthognal
    neg_ortho = gram_schmidt(torch.squeeze(id_emb_target,0))
    neg_ortho = torch.unsqueeze(neg_ortho,0)
    neg_ortho = neg_ortho/torch.norm(neg_ortho, dim=1, keepdim=True)
    id_emb_ortho_template = project_face_embs(pipeline, neg_ortho)    # pass through the encoder
    images_ortho = pipeline(prompt_embeds=id_emb_ortho_template, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images
    print('cosine ortho:',cos(template, neg_ortho))
    for id, img in enumerate(images_ortho):       
        img.save(f"{save_path}/{filename}_{id}_customortho_InsightFace.png")

    # synthetic
    neg_synth = synth_neg(folder_path)
    neg_synth = neg_synth/torch.norm(neg_synth, dim=1, keepdim=True)
    id_emb_synth_template = project_face_embs(pipeline, neg_synth)    # pass through the encoder
    images_synth = pipeline(prompt_embeds=id_emb_synth_template, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images
    print('cosine synth:',cos(template, neg_synth))
    for id, img in enumerate(images_synth):       
        img.save(f"{save_path}/{filename}_{id}_customsynth_InsightFace.png")








