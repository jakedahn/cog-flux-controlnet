import os
import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))


base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
controlnet = FluxControlNetModel.from_pretrained(
    controlnet_model, torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

control_image = load_image("./input-1.png")
prompt = "A girl in city, 25 years old, cool, futuristic"
image = pipe(
    prompt,
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("output.png")
