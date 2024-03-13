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
    pipeline.unload_ip_adapter()

    # embeddings of ip-adapter faceid
    pipeline.load_ip_adapter(
        "IP-Adapter-FaceID",
        subfolder=None,
        weight_name="ip-adapter-faceid_sdxl.bin",
        image_encoder_folder=None
    )
    ref_images = []
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    for im in face_images:
        image = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
        faces = app.get(image)
        image = torch.from_numpy(faces[0].normed_embedding)
        ref_images.append(image.unsqueeze(0))
    ref_images = torch.cat(ref_images, dim=0)

    print(f"InsightFace: {len(face_images)} face images, {len(ref_images)} faces detected")
    assert len(ref_images) > 0, "face detection for faceid failed."

    image_embeds_faceid = pipeline.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[ref_images],
        ip_adapter_image_embeds=None,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
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
    

