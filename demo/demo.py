import gradio as gr

from utils import get_pipe, predict

def run():
    with gr.Blocks() as iface:
        pipe_state = gr.State(value=None)

        models = gr.Dropdown(
            label="Choisir le modèle",
            choices=["diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                     "kandinsky-community/kandinsky-2-2-decoder-inpaint",
                     "stable-diffusion-v1-5/stable-diffusion-inpainting",
                     "stabilityai/stable-diffusion-3.5-large",
                     "black-forest-labs/FLUX.1-schnell"],
            value="stable-diffusion-v1-5/stable-diffusion-inpainting"
        )

        charged_models_button = gr.Button("Charger le modèle")

        # Bouton pour charger le modèle
        charged_models_button.click(
            fn=get_pipe,
            inputs=models,
            outputs=[pipe_state, charged_models_button]
        )

        # Interface principale
        with gr.Row():
            with gr.Column():
                image_editor = gr.ImageEditor(
                    type='numpy',
                    image_mode='RGB',
                    brush=gr.Brush(),
                    label="Editez une image",
                    height=800,
                    width=800
                )
                prompt = gr.Textbox(label="Votre prompt", value="a plaid on a couch")
                steps = gr.Number(label="Nombre de steps", value=30)
                strength = gr.Number(label="Strength", value=1)
                seed = gr.Number(label="Seed", value=1)
                guidance_scale = gr.Number(label="Guidance Scale", value=8)

                predict_button = gr.Button("Générer")
            with gr.Column():
              output_image1 = gr.Image()
              output_image2 = gr.Image()
              output_time = gr.Textbox(label="temps d'inference")

        # Bouton de prédiction
        predict_button.click(
            fn=predict,
            inputs=[models, pipe_state, image_editor, prompt, steps, strength, seed, guidance_scale],
            outputs=[output_image1, output_image2, output_time]
        )

    iface.launch(debug=True, share=True)

if __name__ == "__main__":
    run()