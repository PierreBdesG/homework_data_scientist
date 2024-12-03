import sys
import os

current_dir = os.path.dirname(__file__)
utils_path = os.path.join(current_dir, "ComfyUI")
sys.path.append(utils_path)

import json

import argparse

from utils import pass_argument_to_workflow
from ComfyUI.execution import PromptExecutor
from ComfyUI.server import PromptServer

def execute(workflow):
    e = PromptExecutor(PromptServer(None))
    e.execute(workflow, "", execute_outputs=["9"])

if __name__ == "__main__":
    # Configuration d'argparse
    parser = argparse.ArgumentParser(description="Run a ComfyUI4 workflow")

    # Définir les arguments attendus
    parser.add_argument("--prompt", type=str, default="test", help="The prompt describing the desired image(s).")
    parser.add_argument("--workflow-path", type=str, default="workflows/default_workflow.json", required=False, help="Path to the workflow file.")
    parser.add_argument("--model-name", type=str, default="v1-5-pruned-emaonly.safetensors", help="Name of the model to use.")
    parser.add_argument("--neg-prompt", type=str, default="", help="Negative prompt to specify elements to avoid.")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image in pixels.")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image in pixels.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for random generation.")
    parser.add_argument("--cfg", type=float, default=8.0, help="Classifier-Free Guidance scale.")
    parser.add_argument("--denois", type=float, default=1, help="Denois quantity")
    parser.add_argument("--filename-prefix", type=str, default="ComfyUI4", help="Directory to save the generated images.")
    parser.add_argument("--num-images-per-prompt", type=int, default=1, help="Number of images to generate per prompt.")
    parser.add_argument("--random-seed_after-every-gen", type=bool, default=True, help="Randomize seed after every generation.")
    parser.add_argument("--sampler-name", type=str, default="euler", help="Sampler to use.")
    parser.add_argument("--scheduler", type=str, default="normal", help="Scheduler to use.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda').")

    args = parser.parse_args()
    with open(args.workflow_path, 'r') as file:
        workflow = json.load(file)
    workflow = pass_argument_to_workflow(workflow, args)
    execute(workflow)
