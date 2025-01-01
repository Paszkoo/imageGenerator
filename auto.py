import json
import time
import torch
from PIL import Image
import os
from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5
from transformers import pipeline
from huggingface_hub import login

def load_prompts(json_file):
    """Load prompts from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_models(device=None, model_name="flux-schnell"):
    """Initialize all required models with GPU support."""
    # Force CUDA if available
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    
    torch_device = torch.device(device)
    print(f"Device set to use {device}")
    
    # Move models to GPU explicitly
    t5 = load_t5(torch_device, max_length=256).to(torch_device)
    print("T5 model loaded")
    
    clip = load_clip(torch_device).to(torch_device)
    print("CLIP model loaded")
    
    model = load_flow_model(model_name, device=torch_device).to(torch_device)
    print("Flow model loaded")
    
    ae = load_ae(model_name, device=torch_device).to(torch_device)
    print("AE model loaded")
    
    # Force GPU for NSFW classifier
    nsfw_classifier = pipeline("image-classification", 
                             model="Falconsai/nsfw_image_detection", 
                             device=0 if device == "cuda" else -1)
    print("NSFW classifier loaded")
    
    # Print memory usage if using GPU
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    return model, ae, t5, clip, nsfw_classifier, torch_device

def generate_image(prompt, models, device, 
                  width=1360, height=768, 
                  num_steps=50, guidance=3.5, 
                  seed=None, output_dir="output"):
    """Generate a single image from prompt."""
    model, ae, t5, clip, nsfw_classifier = models[:-1]
    torch_device = device
    
    # Ensure all models are on GPU
    model = model.to(torch_device)
    ae = ae.to(torch_device)
    t5 = t5.to(torch_device)
    clip = clip.to(torch_device)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup sampling options
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed if seed is not None else torch.Generator(device="cpu").seed()
    )

    print(f"Generating '{opts.prompt}' with seed {opts.seed}")
    t0 = time.perf_counter()

    # Prepare input
    x = get_noise(
        1,
        opts.height,
        opts.width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=opts.seed,
    )
    
    timesteps = get_schedule(
        opts.num_steps,
        x.shape[-1] * x.shape[-2] // 4,
        shift=True
    )

    # Prepare and denoise
    with torch.cuda.amp.autocast():  # Enable automatic mixed precision
        inp = prepare(t5=t5, clip=clip, img=x, prompt=opts.prompt)
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

    # Decode to image
    x = unpack(x.float(), opts.height, opts.width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    t1 = time.perf_counter()
    print(f"Done in {t1 - t0:.1f}s.")

    # Convert to PIL Image
    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = x[0].permute(1, 2, 0)
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    # Check NSFW content
    nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
    if nsfw_score >= 0.85:
        print(f"Warning: Image for prompt '{prompt}' may contain NSFW content.")
        return None

    # Clear GPU cache after generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return img

def batch_generate(prompts_file, output_dir="generated_images", model_name="flux-schnell", device=None):
    """Generate images for all prompts in the JSON file."""
    # Load prompts
    prompts_data = load_prompts(prompts_file)
    
    # Setup models
    print("Loading models...")
    models = setup_models(device=device, model_name=model_name)
    print("Models loaded successfully.")

    # Create output directory structure
    base_output_dir = output_dir
    os.makedirs(base_output_dir, exist_ok=True)

    # Generate images for each category
    for category, prompts in prompts_data.items():
        print(f"\nProcessing category: {category}")
        
        # Create category directory
        category_dir = os.path.join(base_output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Generate images for each prompt
        for idx, prompt in enumerate(prompts, 1):
            print(f"\nGenerating image {idx}/{len(prompts)} for {category}")
            print(f"Prompt: {prompt}")
            
            try:
                img = generate_image(prompt, models, models[-1])
                if img is not None:
                    # Save image with prompt as filename (sanitized)
                    safe_prompt = "".join(x for x in prompt[:50] if x.isalnum() or x in (" ", "_", "-"))
                    filename = f"{idx:03d}_{safe_prompt}.jpg"
                    filepath = os.path.join(category_dir, filename)
                    img.save(filepath, format="JPEG", quality=95)
                    print(f"Saved: {filepath}")
                else:
                    print("Image generation failed or NSFW content detected.")
            except Exception as e:
                print(f"Error generating image for prompt: {prompt}")
                print(f"Error: {str(e)}")
                continue

            # Clear GPU cache after each generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Small delay between generations to prevent overload
            time.sleep(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate images from JSON prompts")
    parser.add_argument("--prompts", type=str, required=True, help="Path to JSON prompts file")
    parser.add_argument("--output", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--model", type=str, default="flux-dev", help="Model name")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Device to use for computation")
    parser.add_argument("--token", type=str, help="Hugging Face token", default=None)
    
    args = parser.parse_args()
    
    # Login to Hugging Face if token provided
    if args.token:
        login(token=args.token)
    
    # Set CUDA settings for better performance
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Print CUDA information
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    batch_generate(args.prompts, args.output, args.model, args.device)