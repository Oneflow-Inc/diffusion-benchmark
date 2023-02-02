import os

import argparse
import oneflow as torch
from diffusers import OneFlowStableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--prompt", type=str, default="a photo of an astronaut riding a horse on mars"
    )
    parser.add_argument("--cache", type=str, default="./oneflow-sd/graph_cache")
    parser.add_argument("--model", type=str, default=".//oneflow-sd/model")
    parser.add_argument(
        "--load",
        default=False,
        action="store_true",
        help="If specified, load from cache",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="If specified, save to cache",
    )
    args = parser.parse_args()
    return args


args = parse_args()

model = "CompVis/stable-diffusion-v1-4"
if args.load:
    # Note: restore the cache by setting the pretrain path to a cache path
    model = args.model
    print(f"will load pipe from: {args.cache}")
pipe = OneFlowStableDiffusionPipeline.from_pretrained(
    model,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipe = pipe.to("cuda")
if args.load:
    pipe.load_graph(args.cache)

output_dir = "oneflow-sd-output"
os.makedirs(output_dir, exist_ok=True)

# Note: enable graph-related cache
pipe.set_unet_graphs_cache_size(10)
pipe.enable_graph_share_mem()
pipe.enable_save_graph()


def do_infer(n):
    with torch.autocast("cuda"):
        for i in [2, 1, 0]:
            for j in [2, 1, 0]:
                width = 768 + 128 * i
                height = 768 + 128 * j
                prompt = args.prompt
                images = pipe(prompt, width=width, height=height).images
                for i, image in enumerate(images):
                    prompt = prompt.strip().replace("\n", " ")
                    dst = os.path.join(
                        output_dir, f"{prompt[:100]}-{n}-{width}-{height}.png"
                    )
                    image.save(dst)


for n in range(2):
    do_infer(n)
if args.save:
    print(f"start saving pipe to: {args.cache}")
    os.makedirs(args.cache, exist_ok=True)
    pipe.save_pretrained(args.cache)
    # Note: save graph cache
    pipe.save_graph(args.cache)
