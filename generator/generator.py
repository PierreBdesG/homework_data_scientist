from typing import Union, List
import os
import time

from PIL import Image
from numpy.random import randint
from diffusers import StableDiffusionPipeline
import torch

SAMPLER_NAME = [
        "lms",
        "heun",
        "euler",
        "euler-ancestral",
        "dpm",
        "ddim",
]

class Generator:
    def __init__(self,
                 prompts: Union[List[str], str],
                 model_name: [str],
                 neg_prompts: [Union[List[str], str]],
                 steps: [int],
                 height: [int],
                 width: [int],
                 seed: int,
                 cfg: [float],
                 output_dir: [str],
                 num_images_per_prompt: int,
                 random_seed_after_every_gen: [bool],
                 sampler_name: str,
                 device: str
                 ):
        self.prompts = prompts

        self.neg_prompts = neg_prompts

        self.model_name = model_name

        self.steps = steps
        self.height = height
        self.width = width
        self.num_images_per_prompt = num_images_per_prompt

        if not seed:
            if random_seed_after_every_gen:
                seed = randint(2 ** 32)
            else:
                seed = 0

        self.seed = seed
        self.cfg = cfg
        self.output_dir = output_dir

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device not in ["cuda", "cpu"]:
            raise ValueError(f"device must be either 'cuda' or 'cpu', not {device}")

        if not output_dir:
            output_dir = "output_images"
        self.output_dir = output_dir

        if sampler_name not in SAMPLER_NAME:
            raise ValueError(f"sampler_name must be one of {SAMPLER_NAME} not {sampler_name}")
        self.sampler_name = sampler_name

        self.device = device

        self.pipe: StableDiffusionPipeline = None
        self.images: List[Image] = []


    def load_model(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_name).to(self.device)
        self.pipe.sampler_name = self.sampler_name

    def predict(self):
        with torch.no_grad():
            self.images = self.pipe(
                prompt = self.prompts,
                height = self.height,
                width = self.width,
                num_inference_steps = self.steps,
                guidance_scale = self.cfg,
                negative_prompt = self.neg_prompts,
                num_images_per_prompt = self.num_images_per_prompt,
                generator = torch.Generator(device=self.device).manual_seed(self.seed),
            ).images

    def save_images(self):
        os.makedirs(self.output_dir, exist_ok=True)

        for img in self.images:
            full_path = os.path.join(self.output_dir, f"{str(time.time())}.png")
            img.save(full_path)

    def run(self):
        self.load_model()
        self.predict()
        self.save_images()
