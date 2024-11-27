import gradio as gr
from diffusers import AutoPipelineForInpainting
import torch
from PIL import Image
import time

def get_pipe(model_name):
    torch.cuda.empty_cache()
    pipe = AutoPipelineForInpainting.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe, gr.Button(value="Modèle Chargé", interactive=False)

def get_mask(image):
    mask = image["layers"][0]
    mask[mask > 0] = 255
    return Image.fromarray(mask)

def predict(model, pipe, image_data, prompt, nb_step, strength, seed, guidance_scale):
    start = time.time()
    if pipe is None:
        raise gr.Error("Veuillez charger un modèle avant de générer")

    image = Image.fromarray(image_data["background"])
    mask = get_mask(image_data)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    image_gen = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,  # Utilisez le paramètre passé
        num_inference_steps=nb_step,
        strength=strength,
        generator=generator,
    ).images[0]

    return image_gen.resize(image.size), Image.fromarray(image_data["composite"]), time.time() - start
