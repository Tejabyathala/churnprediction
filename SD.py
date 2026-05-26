import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from datetime import datetime

# Configuration
MODEL_ID = "stabilityai/stable-diffusion-2-1"  # Official Stable Diffusion 2.1
SAVE_DIR = "sd_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEIGHT = 512  # Stable Diffusion's default resolution
WIDTH = 512

def generate_image(prompt: str) -> Image.Image:
    """Generate image using Stable Diffusion"""
    # Load pipeline with safety checker
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None  # Optional: Remove for NSFW filtering
    ).to(DEVICE)
    
    return pipe(
        prompt=prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=50,  # Optimal for quality/speed balance
        guidance_scale=7.5,  # Standard CFG scale
        generator=None
    ).images[0]

def save_image(image: Image.Image, prompt: str) -> str:
    """Save image with prompt-based filename"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"sd_{timestamp}_{prompt[:30].replace(' ', '_')}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    image.save(save_path)
    return save_path

def main():
    try:
        prompt = input("\nEnter your image description: ")
        print("\nGenerating image... (This may take 10-30 seconds)")
        
        image = generate_image(prompt)
        save_path = save_image(image, prompt)
        
        print(f"\n✅ Image saved to: {save_path}")
        image.show()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    print("=== Stable Diffusion Image Generator ===")
    main()