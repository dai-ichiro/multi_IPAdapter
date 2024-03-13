import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import argparse
import itertools
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="sdxl model",
)
parser.add_argument(
    "--plusface",
    type=str,
    required=True,
    help="embeddings for plusface"
)
parser.add_argument(
    "--plusface_scale",
    type=float,
    help="scale of ip-apdapter plusface"
)
parser.add_argument(
    "--faceid",
    type=str,
    required=True,
    help="embeddings for facdid"
)
parser.add_argument(
    "--faceid_scale",
    type=float,
    help="scale of ip-apdapter faceid"
)
parser.add_argument(
    "--style",
    type=str,
    help="embeddings for style",
)
parser.add_argument(
    "--style_scale",
    type=float,
    help="scale of style"
)
parser.add_argument(
    "--prompt",
    type=str,
    default="a woman",
    help="prompt"
)
parser.add_argument(
    "--negative_prompt",
    type=str,
    default="monochrome, lowres, bad anatomy, worst quality, low quality",
    help="negative prompt"
)
args = parser.parse_args()

pipeline = AutoPipelineForText2Image.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True
)

prompt = args.prompt
negative_prompt = args.negative_prompt

plusface_scale_list = [args.plusface_scale] if args.plusface_scale else [0.1, 0.3, 0.5]
faceid_scale_list = [args.faceid_scale] if args.faceid_scale else [0.3, 0.5, 0.7, 0.9]

if args.style:
    save_folder = "results_with_style"
    Path(save_folder).mkdir(exist_ok=True)
    style_scale_list = [args.style_scale] if args.style_scale else [0.3, 0.5]
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

    image_embeds_plusface = torch.load(args.plusface)
    image_embeds_faceid = torch.load(args.faceid)
    image_embeds_style = torch.load(args.style)
    image_embeds = [image_embeds_style[0], image_embeds_plusface[0], image_embeds_faceid[0]]

    for (style_scale, plusface_scale, faceid_scale) in itertools.product(style_scale_list, plusface_scale_list, faceid_scale_list):
        pipeline.set_ip_adapter_scale([style_scale, plusface_scale, faceid_scale])
        image = pipeline(
            prompt=prompt,
            ip_adapter_image_embeds=image_embeds,
            negative_prompt=negative_prompt, 
            num_inference_steps=50,
            num_images_per_prompt=1,
            guidance_scale = 7.5,
            width=1024,
            height=1024, 
            generator=torch.Generator(device="cpu").manual_seed(0)
        ).images[0]

        save_fname = f"plusface{plusface_scale}_faceid{faceid_scale}_with_style{style_scale}.png"
        image.save(Path(save_folder, save_fname).as_posix())

else:
    save_folder = "results_without_style"
    Path(save_folder).mkdir(exist_ok=True)
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

    image_embeds_plusface = torch.load(args.plusface)
    image_embeds_faceid = torch.load(args.faceid)
    image_embeds = [image_embeds_plusface[0], image_embeds_faceid[0]]

    for (plusface_scale, faceid_scale) in itertools.product(plusface_scale_list, faceid_scale_list):
        pipeline.set_ip_adapter_scale([plusface_scale, faceid_scale])
        image = pipeline(
            prompt=prompt,
            ip_adapter_image_embeds=image_embeds,
            negative_prompt=negative_prompt, 
            num_inference_steps=50,
            num_images_per_prompt=1,
            guidance_scale = 7.5,
            width=1024,
            height=1024, 
            generator=torch.Generator(device="cpu").manual_seed(0)
        ).images[0]

        save_fname = f"plusface{plusface_scale}_faceid{faceid_scale}.png"
        image.save(Path(save_folder, save_fname).as_posix())
