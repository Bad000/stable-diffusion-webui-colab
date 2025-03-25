import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

# Загружаем модель
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

def generate(prompt):
    image = pipeline(prompt).images[0]
    return image

iface = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Введите запрос для генерации изображения"),
    outputs=gr.Image(label="Сгенерированное изображение"),
    title="Stable Diffusion WebUI",
    description="Простое веб-приложение для генерации изображений с помощью Stable Diffusion"
)

iface.launch(share=True)
