import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    required=True,
    help="sdxl model",
)
parser.add_argument(
    '--plusface',
    type=str,
    required=True,
    help="embeddings for plusface"
)
parser.add_argument(
    '--faceid',
    type=str,
    required=True,
    help="embeddings for facdid"
)
parser.add_argument(
    '--style',
    type=str,
    help="embeddings for style",
)
args = parser.parse_args()

model_id = args.model

pipeline = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True
)
if args.style:
    # text2image with 3 image embeddings
    pipeline.load_ip_adapter(
        ["IP-Adapter", "IP-Adapter", "IP-Adapter-FaceID"],
        subfolder=["sdxl_models", "sdxl_models", None],
        weight_name=[
            "ip-adapter-plus_sdxl_vit-h.safetensors",
            "ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "ip-adapter-faceid_sdxl.bin"
        ],
        image_encoder_folder=None
    )
    pipeline.set_ip_adapter_scale([0.5, 0.5, 0.5])

    image_embeds_plusface = torch.load(args.plusface)
    image_embeds_faceid = torch.load(args.faceid)
    image_embeds_style = torch.load(args.style)
    image_embeds = [image_embeds_style[0], image_embeds_plusface[0], image_embeds_faceid[0]]

    image = pipeline(
        prompt="a woman",
        ip_adapter_image_embeds=image_embeds,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
        num_inference_steps=50,
        num_images_per_prompt=1,
        guidance_scale = 7.5,
        width=1024,
        height=1024, 
        generator=torch.Generator(device="cpu").manual_seed(0)
    ).images[0]

    image.save("plusface_faceid_with_style.png")

else:
    # text2image with 2 image embeddings
    pipeline.load_ip_adapter(
        ["IP-Adapter", "IP-Adapter-FaceID"],
        subfolder=["sdxl_models", None],
        weight_name=[
            "ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "ip-adapter-faceid_sdxl.bin"
        ],
        image_encoder_folder=None
    )
    pipeline.set_ip_adapter_scale([0.5, 0.5])

    image_embeds_plusface = torch.load(args.plusface)
    image_embeds_faceid = torch.load(args.faceid)
    image_embeds = [image_embeds_plusface[0], image_embeds_faceid[0]]

    image = pipeline(
        prompt="a woman",
        ip_adapter_image_embeds=image_embeds,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
        num_inference_steps=50,
        num_images_per_prompt=1,
        guidance_scale = 7.5,
        width=1024,
        height=1024, 
        generator=torch.Generator(device="cpu").manual_seed(0)
    ).images[0]

    image.save("plusface_faceid.png")
