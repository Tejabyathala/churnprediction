import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os
from datetime import datetime

# Configuration
MODEL_ID = "CompVis/ldm-text2im-large-256"
SAVE_DIR = "ldm_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_image(prompt: str) -> Image.Image:
    """Generate image using LDM with default parameters"""
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        custom_pipeline="latent_diffusion/unconditional",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    return pipe(
        prompt=prompt,
        height=256,  # Fixed resolution
        width=256,
        num_inference_steps=100,  # Default steps
        guidance_scale=3.0,  # Default guidance
        generator=None  # Random seed
    ).images[0]

def save_image(image: Image.Image, prompt: str) -> str:
    """Save image with prompt-based filename"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{prompt[:30].replace(' ', '_')}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    image.save(save_path)
    return save_path

def main():
    try:
        prompt = input("\nEnter your image description: ")
        print("\nGenerating image... (This may take a moment)")
        
        image = generate_image(prompt)
        save_path = save_image(image, prompt)
        
        print(f"\nImage ready! Saved to: {save_path}")
        image.show()
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    print("=== LDM Image Generator ===")
    main()