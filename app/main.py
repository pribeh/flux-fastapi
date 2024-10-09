import io
import uuid
import base64
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

app = FastAPI()

# Initialize the model with mixed precision (FP16)
device = "cuda" if torch.cuda.is_available() else "cpu"
flux_model = StableDiffusionPipeline.from_pretrained(
    "/models/flux/schnell", 
    torch_dtype=torch.float16  # Enable mixed precision for performance
).to(device)

# Default model parameters
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 2.0

# Request body model
class RequestRunPod(BaseModel):
    api_name: str
    prompt: str
    image_prompts: list[str] = []  # Base64-encoded images
    style_selections: list = []
    negative_prompt: str = ""
    aspect_ratios_selection: str = "1024*1024"
    image_number: int = 1
    image_seed: int = -1
    steps: int = DEFAULT_STEPS  # Default steps to 4, can be overridden
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE  # Default guidance scale to 2

# Function to convert image to base64
def image_to_base64(image, format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.post("/runpod")
async def runpod_api(request: RequestRunPod):
    try:
        # Handle img2img with multiple image prompts combined
        if request.api_name == "img2img2":
            if not request.image_prompts:
                raise HTTPException(status_code=400, detail="Image prompts are required for img2img.")
            
            # Decode and process all images in the image_prompts list
            input_images = [Image.open(io.BytesIO(base64.b64decode(img))) for img in request.image_prompts]

            # Here, we can combine the influence of all images in a single generation
            # By default, the first image could be the main guide, and others provide additional context.
            # Alternatively, you can experiment with averaging, blending, or using them as latent conditions.
            output_image = flux_model(
                prompt=request.prompt,
                image=input_images,  # Pass the list of images to be combined as conditioning
                guidance_scale=request.guidance_scale, 
                num_inference_steps=request.steps
            ).images[0]

            # Convert the output image to base64
            base64_image = image_to_base64(output_image)

            return {
                "status": "success",
                "asset_id": str(uuid.uuid4()),  # Generate a unique asset ID
                "image_base64": base64_image
            }

        elif request.api_name == "txt2img":
            # Handle txt2img as before
            output_images = flux_model(
                prompt=request.prompt, 
                guidance_scale=request.guidance_scale, 
                num_inference_steps=request.steps
            ).images
            
            base64_image = image_to_base64(output_images[0])
            return {
                "status": "success",
                "asset_id": str(uuid.uuid4()),  # Generate a unique asset ID
                "image_base64": base64_image
            }

        else:
            raise HTTPException(status_code=400, detail="Invalid API name. Must be 'img2img2' or 'txt2img'.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
