from typing import Union, List, Optional

import fire

from generator.generator import Generator

def run(prompt: Union[List[str], str],
        model_name: Optional[str] = "runwayml/stable-diffusion-v1-5",
        neg_prompt: Optional[Union[List[str], str]] = "",
        steps: Optional[int] = 50,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        seed: int = None,
        cfg: Optional[float] = 8,
        output_dir: Optional[str] = "",
        num_images_per_prompt: Optional[int] = 1,
        random_seed_after_every_gen: Optional[bool] = True,
        sampler_name: Optional[str] = "normal",
        device: Optional[str] = None):

    generator = Generator(
        prompts = prompt,
        model_name = model_name,
        neg_prompts = neg_prompt,
        steps = steps,
        height = height,
        width = width,
        seed = seed,
        cfg = cfg,
        output_dir = output_dir,
        num_images_per_prompt = num_images_per_prompt,
        random_seed_after_every_gen = random_seed_after_every_gen,
        sampler_name = sampler_name,
        device = device
    )

    generator.run()

if __name__ == "__main__":
    fire.Fire(run)
