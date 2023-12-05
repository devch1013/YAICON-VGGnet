import os
import imageio
from PIL import Image

import torch
import torch.nn.functional as F

from diffusers import IFSuperResolutionPipeline, VideoToVideoSDPipeline
from diffusers.utils.torch_utils import randn_tensor

import sys

sys.path.append("root directory of project")

from embedding.models.imagebind import ImageBindEmbedding
from embedding.trainer.models.mapping_model import BindNetwork
from ImageBind.imagebind.models.imagebind_model import ModalityType


from showone.pipelines import (
    TextToVideoIFPipeline,
    TextToVideoIFInterpPipeline,
    TextToVideoIFSuperResolutionPipeline,
)
from showone.pipelines.pipeline_t2v_base_pixel import tensor2vid
from showone.pipelines.pipeline_t2v_sr_pixel_cond import TextToVideoIFSuperResolutionPipeline_Cond


# Base Model
# Only use first stage to get imagebind result
pretrained_model_path = "show_1/weights/show-1-base"
pipe_base = TextToVideoIFPipeline.from_pretrained(
    pretrained_model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    local_files_only=True
    # force_download=True
)
pipe_base.enable_model_cpu_offload()


def type_parser(prompt_type):
    if prompt_type == "text":
        return ModalityType.TEXT
    if prompt_type == "audio":
        return ModalityType.AUDIO
    if prompt_type == "vision":
        return ModalityType.VISION


# Inference
prompt = [ # put your test promts or directory of audio or image files
    "Cute black cat in the summer",
    "Young rabbit eating grass in the forest.",
    "Running ambulance with sirens blaring",
    "Trees in the wind",
    "Airplane flying at cruising altitude, condense trail remaining. aka contrail or chemtrail.",
    "Retro toy car",
    "Fireworks in venice",
]
prefix = "Inference_test_"
prompt_type = "text" # or audio or image
output_dir = "./outputs/example"
negative_prompt = "low resolution, blur"

seed = 345
os.makedirs(output_dir, exist_ok=True)

imagebind_model = ImageBindEmbedding()
vgg_model = BindNetwork()
# new_state_dict = OrderedDict()
state_dict = torch.load(
    "models/VGGnet_final_weight.pth"
)
vgg_model.load_state_dict(state_dict)
vgg_model = vgg_model.to("cuda")
vgg_model.eval()

for prom in prompt:
    image_name = prefix + prom.split("/")[-1].split(".")[0]
    with torch.no_grad():
        prompt_embeds_ib = imagebind_model([prom], prompt_type)[type_parser(prompt_type)]
        print("iamgebind embedding: ", prompt_embeds_ib)
        prompt_embeds, _ = vgg_model(prompt_embeds_ib)

    prompt_embeds, negative_embeds = pipe_base.encode_prompt(
        prompt=None, negative_prompt=negative_prompt, prompt_embeds=prompt_embeds
    )

    video_frames = pipe_base(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        num_frames=8,
        height=40,
        width=64,
        num_inference_steps=75,
        guidance_scale=9.0,
        generator=torch.manual_seed(seed),
        output_type="pt",
    ).frames

    print("save base gif: ", image_name)
    imageio.mimsave(f"{output_dir}/{image_name}_base.gif", tensor2vid(video_frames.clone()), fps=2)
