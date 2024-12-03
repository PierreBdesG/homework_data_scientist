from typing import Dict, Union, List


def get_nodes_from_class_type(workflow, class_type):
    for key, value in workflow.items():
        current_class_type = value.get('class_type')
        if current_class_type == class_type:
            return key
    return None


def pass_argument_to_workflow(workflow: Dict, args) -> Union[Dict, List[str]]:
    sampler_node = get_nodes_from_class_type(workflow, "KSampler")
    workflow[sampler_node]["inputs"]["seed"] = args.seed
    workflow[sampler_node]["inputs"]["steps"] = args.steps
    workflow[sampler_node]["inputs"]["cfg"] = args.cfg
    workflow[sampler_node]["inputs"]["sampler_name"] = args.sampler_name
    workflow[sampler_node]["inputs"]["scheduler"] = args.scheduler
    workflow[sampler_node]["inputs"]["denois"] = args.denois

    prompt_node = workflow[sampler_node]["inputs"]["positive"][0]
    workflow[prompt_node]["inputs"]["text"] = args.prompt

    neg_prompt_node = workflow[sampler_node]["inputs"]["negative"][0]
    workflow[neg_prompt_node]["inputs"]["text"] = args.neg_prompt

    model_node = get_nodes_from_class_type(workflow, "CheckpointLoaderSimple")
    workflow[model_node]["inputs"]["ckpt_name"] = args.model_name

    latent_node = get_nodes_from_class_type(workflow, "EmptyLatentImage")
    workflow[latent_node]["inputs"]["height"] = args.height
    workflow[latent_node]["inputs"]["width"] = args.width
    workflow[latent_node]["inputs"]["batch_size"] = args.num_images_per_prompt

    save_node = get_nodes_from_class_type(workflow, "SaveImage")
    workflow[save_node]["inputs"]["filename_prefix"] = args.filename_prefix
    return workflow, [save_node]