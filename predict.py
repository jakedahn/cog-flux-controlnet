import os
import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from cog import BasePredictor, Input, Path
from huggingface_hub import login


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        control_image: Path = Input(description="Input image for control"),
        prompt: str = Input(description="Text prompt for image generation"),
        controlnet_conditioning_scale: float = Input(
            description="ControlNet conditioning scale", default=0.6
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps", default=28
        ),
        guidance_scale: float = Input(description="Guidance scale", default=3.5),
        hf_token: str = Input(description="Hugging Face API token", default=None),
    ) -> Path:
        if hf_token:
            login(token=hf_token)

        base_model = "black-forest-labs/FLUX.1-dev"
        controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch.bfloat16
        )
        self.pipe = FluxControlNetPipeline.from_pretrained(
            base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")

        control_image = load_image(str(control_image))
        image = self.pipe(
            prompt,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        output_path = Path("output.png")
        image.save(str(output_path))
        return output_path
