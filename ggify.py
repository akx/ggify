import argparse
import os
import re
import subprocess
import sys

import huggingface_hub
import tqdm
from huggingface_hub.hf_api import RepoFile

hf_token = huggingface_hub.get_token()

KNOWN_QUANTIZATION_TYPES = {
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q2_k",
    "q3_k_s",
    "q3_k_m",
    "q3_k_l",
    "q4_k_s",
    "q4_k_m",
    "q5_k_s",
    "q5_k_m",
    "q6_k",
    "q8_0",
}


def get_llama_cpp_dir():
    dir = os.environ.get("LLAMA_CPP_DIR", "./")
    if not os.path.isdir(dir):
        raise ValueError(f"Could not find llama.cpp directory at {dir}")
    return dir


PYTHON_EXE = os.environ.get("PYTHON_EXE", sys.executable)
GG_MODEL_EXTENSION = ".gguf"


def quantize_f32(dirname, type: str) -> str:
    q_model_path = os.path.join(dirname, f"ggml-model-{type}{GG_MODEL_EXTENSION}")
    f32_model_path = os.path.join(dirname, f"ggml-model-f32{GG_MODEL_EXTENSION}")
    if not os.path.isfile(q_model_path):
        if not f32_model_path:
            raise ValueError(f"Could not find fp32 model at {f32_model_path}")
        quantize_cmd = os.path.join(get_llama_cpp_dir(), "quantize")

        if not os.path.isfile(quantize_cmd):
            raise RuntimeError(
                f"Could not find quantize executable at {quantize_cmd} "
                f"(set LLAMA_CPP_DIR (currently {get_llama_cpp_dir()}?))"
            )
        concurrency = str(os.cpu_count() + 2)
        subprocess.check_call([quantize_cmd, f32_model_path, type, concurrency])
    return q_model_path


def get_ggml_model_path(dirname: str, convert_type: str):
    if convert_type in ("0", "f32"):
        type_moniker = "f32"
    elif convert_type == ("1", "f16"):
        type_moniker = "f16"
    else:
        raise ValueError(f"Unknown type {convert_type}")
    model_path = os.path.join(dirname, f"ggml-model-{type_moniker}{GG_MODEL_EXTENSION}")
    return model_path


def convert_pth(dirname, *, convert_type: str, vocab_type: str):
    model_path = get_ggml_model_path(dirname, convert_type)
    if not os.path.isfile(model_path):
        convert_py = os.path.join(get_llama_cpp_dir(), "convert.py")
        if not os.path.isfile(convert_py):
            raise RuntimeError(
                f"Could not find convert.py at {convert_py} "
                f"(set LLAMA_CPP_DIR (currently {get_llama_cpp_dir()}?))"
            )
        command = [
            PYTHON_EXE,
            convert_py,
            dirname,
            f"--outtype={convert_type}",
            f"--vocab-type={vocab_type}",
        ]
        subprocess.check_call(command)
    return model_path


def convert_pth_to_types(dirname, *, types, remove_f32_model=False, vocab_type: str):
    # If f32 is requested, or a quantized type is requested, convert to fp32 GGML
    f32_path = None
    if "f32" in types or any(t.startswith("q") for t in types):
        f32_path = convert_pth(dirname, convert_type="f32", vocab_type=vocab_type)
    # Other types
    for type in types:
        if type.startswith("q"):
            q_model_path = quantize_f32(dirname, type=type.upper())
            yield q_model_path
        elif type == "f16":
            yield convert_pth(dirname, convert_type="f16")
        elif type == "f32":
            pass  # already dealt with
        else:
            raise ValueError(f"Unknown type {type}")
    if "f32" not in types and remove_f32_model:
        f32_model_path = get_ggml_model_path(dirname, "f32")
        print(f"Removing fp32 model {f32_model_path}")
        os.remove(f32_model_path)
    elif f32_path:
        yield f32_path


def download_repo(repo, dirname):
    files = list(huggingface_hub.list_files_info(repo, token=hf_token))
    if not any(fi.rfilename.startswith("pytorch_model") for fi in files):
        print(
            f"Repo {repo} does not seem to contain a PyTorch model, but continuing anyway"
        )

    with tqdm.tqdm(files, unit="file", desc="Downloading files...") as pbar:
        fileinfo: RepoFile
        for fileinfo in pbar:
            filename = fileinfo.rfilename
            basename = os.path.basename(filename)
            if basename.startswith("."):
                continue
            if basename.endswith(".gguf"):
                continue
            if os.path.isfile(os.path.join(dirname, filename)):
                continue
            pbar.set_description(f"{filename} ({fileinfo.size // 1048576:d} MB)")
            huggingface_hub.hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=dirname,
                token=hf_token,
            )


def main():
    quants = ",".join(KNOWN_QUANTIZATION_TYPES)
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=str, help="Huggingface repository to convert")
    ap.add_argument(
        "--types",
        "-t",
        type=str,
        help=f"Quantization types, comma-separated (default: %(default)s; available: f16,f32,{quants})",
        default="q4_0,q4_1,q8_0",
    )
    ap.add_argument(
        "--llama-cpp-dir",
        type=str,
        help="Directory containing llama.cpp (default: %(default)s)",
        default=get_llama_cpp_dir(),
    )
    ap.add_argument(
        "--keep-f32-model",
        action="store_true",
        help="Don't remove the fp32 model after quantization (unless it's explicitly requested)",
    )
    ap.add_argument(
        "--vocab-type",
        type=str,
        default="spm",
    )
    args = ap.parse_args()
    if args.llama_cpp_dir:
        os.environ["LLAMA_CPP_DIR"] = args.llama_cpp_dir
    repo = args.repo
    dirname = os.path.join(".", "models", repo.replace("/", "__"))
    download_repo(repo, dirname)
    types = set(re.split(r",\s*", args.types))
    output_paths = list(
        convert_pth_to_types(
            dirname,
            types=types,
            remove_f32_model=not args.keep_f32_model,
            vocab_type=args.vocab_type,
        )
    )
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
