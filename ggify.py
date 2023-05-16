import os
import re
import sys

import tqdm
from huggingface_hub.hf_api import RepoFile
import huggingface_hub
import argparse
import subprocess

LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", "./")
QUANTIZE = os.path.join(LLAMA_CPP_DIR, "quantize")
CONVERT_PTH = os.path.join(LLAMA_CPP_DIR, "convert-pth-to-ggml.py")
PYTHON_EXE = os.environ.get("PYTHON_EXE", sys.executable)


def quantize_f32(dirname, type):
    q_model_path = os.path.join(dirname, f"ggml-model-{type}.bin")
    f32_model_path = os.path.join(dirname, "ggml-model-f32.bin")
    if not os.path.isfile(q_model_path):
        if not f32_model_path:
            raise ValueError(f"Could not find fp32 model at {f32_model_path}")
        subprocess.check_call([QUANTIZE, f32_model_path, type])
    return q_model_path


def convert_pth(dirname, *, type: str):
    if type == "0":
        type_moniker = "f32"
    elif type == "1":
        type_moniker = "f16"
    else:
        raise ValueError(f"Unknown type {type}")
    model_path = os.path.join(dirname, f"ggml-model-{type_moniker}.bin")
    if not os.path.isfile(model_path):
        subprocess.check_call([PYTHON_EXE, CONVERT_PTH, dirname, type])
    return model_path


def convert_pth_to_types(dirname, types):
    # If f32 is requested, or a quantized type is requested, convert to fp32 GGML
    if "f32" in types or any(t.startswith("q") for t in types):
        yield convert_pth(dirname, type="0")
    # Other types
    for type in types:
        if type.startswith("q"):
            q_model_path = quantize_f32(dirname, type)
            yield q_model_path
        elif type == "f16":
            yield convert_pth(dirname, type="1")
        elif type == "f32":
            pass  # already dealt with
        else:
            raise ValueError(f"Unknown type {type}")


def download_repo(repo, dirname):
    files = list(huggingface_hub.list_files_info(repo))
    if not any(fi.rfilename.startswith("pytorch_model-") for fi in files):
        raise ValueError(f"Repo {repo} does not contain a PyTorch model")

    with tqdm.tqdm(files, unit="file", desc="Downloading files...") as pbar:
        fileinfo: RepoFile
        for fileinfo in pbar:
            filename = fileinfo.rfilename
            if os.path.basename(filename).startswith("."):
                continue
            if os.path.isfile(os.path.join(dirname, filename)):
                continue
            pbar.set_description(f"{filename} ({fileinfo.size // 1048576:d} MB)")
            huggingface_hub.hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=dirname,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=str, help="Huggingface repository to convert")
    ap.add_argument(
        "--types",
        "-t",
        type=str,
        help="Quantization types, comma-separated (default: %(DEFAULT)s; available: f16,f32,q4_0,q4_1,q5_0,q5_1,q8_0)",
        default="f32,q4_0,q4_1,q8_0",
    )
    args = ap.parse_args()

    if not os.path.isdir(LLAMA_CPP_DIR):
        ap.error(f"Could not find llama.cpp directory at {LLAMA_CPP_DIR}")

    if not os.path.isfile(QUANTIZE):
        ap.error(f"Could not find quantize executable at {QUANTIZE}")

    repo = args.repo
    dirname = os.path.join(".", "models", repo.replace("/", "__"))
    download_repo(repo, dirname)
    types = set(re.split(r",\s*", args.types))
    output_paths = list(convert_pth_to_types(dirname, types=types))
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
