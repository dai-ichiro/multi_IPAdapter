import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
from pathlib import Path
import argparse

def cref_embeddings(pipeline, folder):

    from insightface.app import FaceAnalysis
    import cv2
    import numpy as np

    face_images = [load_image(x.as_posix()) for x in Path(folder).glob("*.png")]

    # embeddings of ip-adapter plus face
    pipeline.load_ip_adapter(
        "IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        image_encoder_folder="models/image_encoder"
    )
    image_embeds_plusface = pipeline.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[face_images],
        ip_adapter_image_embeds=None,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True
    )
    torch.save(image_embeds_plusface, "plusface.ipadpt")

    # embeddings of ip-adapter faceid
    ref_images_embeds = []
    ref_unc_images_embeds = []
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    for im in face_images:
        image = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
        faces = app.get(image)
        if len(faces) > 0:
            image = torch.from_numpy(faces[0].normed_embedding)
            image_embeds = image.unsqueeze(0)
            uncond_image_embeds = torch.zeros_like(image_embeds)
            ref_images_embeds.append(image_embeds)
            ref_unc_images_embeds.append(uncond_image_embeds)

    print(f"InsightFace: {len(face_images)} face images, {len(ref_images_embeds)} faces detected")
    assert len(ref_images_embeds) > 0, "face detection for faceid failed."
    
    ref_images_embeds = torch.stack(ref_images_embeds, dim=0)
    ref_unc_images_embeds = torch.stack(ref_unc_images_embeds, dim=0)
    image_embeds_faceid = [torch.cat([ref_unc_images_embeds, ref_images_embeds], dim=0).to(device="cuda", dtype=torch.float16)]

    torch.save(image_embeds_faceid, "faceid.ipadpt")

def sref_embeddings(pipeline, folder):

    style_images = [load_image(x.as_posix()) for x in Path(folder).glob("*.png")]

    # embeddings of ip-adapter style
    pipeline.load_ip_adapter(
        "IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        image_encoder_folder="models/image_encoder"
    )
    image_embeds_style = pipeline.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[style_images],
        ip_adapter_image_embeds=None,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True
    )
    torch.save(image_embeds_style, "style.ipadpt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cref", type=str, help="folder of character reference")
    parser.add_argument("--sref", type=str, help="folder of style reference")
    args = parser.parse_args()

    pipeline = AutoPipelineForText2Image.from_pretrained(
        "model/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    if args.cref:
        cref_embeddings(pipeline, args.cref)
    if args.sref:
        sref_embeddings(pipeline, args.sref)
    

