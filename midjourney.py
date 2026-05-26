import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import argparse
import os
from datetime import datetime
from typing import Optional

# Configuration
MODEL_CHOICES = {
    "v1-5": "sd-legacy/stable-diffusion-v1-5",
    "xl": "stabilityai/stable-diffusion-xl-base-1.0"
}
SAVE_DIR = "generated_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def generate_image(
    prompt: str,
    model_type: str = "v1-5",
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate image using specified Stable Diffusion model
    """
    # Model-specific validation
    if model_type == "v1-5":
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("For v1-5, height/width must be divisible by 8")
    elif model_type == "xl":
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError("For XL, height/width must be divisible by 64")
    
    # Load appropriate pipeline
    if model_type == "xl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_CHOICES[model_type],
            torch_dtype=TORCH_DTYPE
        ).to(DEVICE)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_CHOICES[model_type],
            torch_dtype=TORCH_DTYPE
        ).to(DEVICE)
    
    # Generation parameters
    generator = torch.Generator(device=DEVICE).manual_seed(seed) if seed else None
    
    with torch.autocast(DEVICE):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator
        )
    
    return result.images[0]

def save_image(image: torch.Tensor, prompt: str, model_type: str) -> str:
    """
    Save generated image with metadata in filename
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{model_type}_{prompt[:40].replace(' ', '_')}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    image.save(save_path)
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generator")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--model", choices=["v1-5", "xl"], default="v1-5",
                       help="Model version (v1-5 or xl)")
    parser.add_argument("--negative", type=str, default=None,
                       help="Negative prompt")
    parser.add_argument("--steps", type=int, default=50,
                       help="Inference steps (50 for v1-5, 70 for XL)")
    parser.add_argument("--guidance", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--height", type=int, default=512,
                       help="Image height (512 for v1-5, 1024 for XL)")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    
    args = parser.parse_args()
    
    try:
        print(f"Generating image using {args.model.upper()} model...")
        image = generate_image(
            prompt=args.prompt,
            model_type=args.model,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed
        )
        
        save_path = save_image(image, args.prompt, args.model)
        print(f"Image saved to: {save_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()