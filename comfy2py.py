from typing import Union, List, Optional

import argparse

from generator.generator import Generator

def run(prompt: Union[List[str], str],
        model_name: Optional[str] = "runwayml/stable-diffusion-v1-5",
        neg_prompt: Optional[Union[List[str], str]] = "",
        steps: Optional[int] = 50,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        seed: int = None,
        cfg: Optional[float] = 8,
        output_dir: Optional[str] = "",
        num_images_per_prompt: Optional[int] = 1,
        random_seed_after_every_gen: Optional[bool] = True,
        sampler_name: Optional[str] = "euler",
        scheduler: Optional[str] = "normal",
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
        scheduler = scheduler,
        device = device
    )

    generator.run()

if __name__ == "__main__":
    # Configuration d'argparse
    parser = argparse.ArgumentParser(description="Run a stable diffusion image generator.")

    # Définir les arguments attendus
    parser.add_argument("--prompt", type=str, required=True, help="The prompt describing the desired image(s).")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Name of the model to use.")
    parser.add_argument("--neg_prompt", type=str, default="", help="Negative prompt to specify elements to avoid.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image in pixels.")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image in pixels.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generation.")
    parser.add_argument("--cfg", type=float, default=8.0, help="Classifier-Free Guidance scale.")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save the generated images.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt.")
    parser.add_argument("--random_seed_after_every_gen", type=bool, default=True, help="Randomize seed after every generation.")
    parser.add_argument("--sampler_name", type=str, default="euler", help="Sampler to use.")
    parser.add_argument("--scheduler", type=str, default="normal", help="Scheduler to use.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda').")

    # Parse les arguments
    args = parser.parse_args()

    # Appel de la fonction `run` avec les arguments parsés
    run(
        prompt=args.prompt,
        model_name=args.model_name,
        neg_prompt=args.neg_prompt,
        steps=args.steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
        cfg=args.cfg,
        output_dir=args.output_dir,
        num_images_per_prompt=args.num_images_per_prompt,
        random_seed_after_every_gen=args.random_seed_after_every_gen,
        sampler_name=args.sampler_name,
        scheduler=args.scheduler,
        device=args.device
    )