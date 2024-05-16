import argparse
import os
import re
import shlex
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


def print_and_check_call(args: list):
    print("=> Running:", shlex.join(args))
    return subprocess.check_call(args)


def quantize(
    dirname,
    *,
    src_type: str,
    dest_type: str,
) -> str:
    q_model_path = os.path.join(
        dirname,
        f"ggml-model-{dest_type}{GG_MODEL_EXTENSION}",
    )
    nonq_model_path = os.path.join(
        dirname,
        f"ggml-model-{src_type}{GG_MODEL_EXTENSION}",
    )
    if not os.path.isfile(q_model_path):
        if not nonq_model_path:
            raise ValueError(f"Could not find nonquantized model at {nonq_model_path}")
        quantize_cmd = os.path.join(get_llama_cpp_dir(), "quantize")

        if not os.path.isfile(quantize_cmd):
            raise RuntimeError(
                f"Could not find quantize executable at {quantize_cmd} "
                f"(set LLAMA_CPP_DIR (currently {get_llama_cpp_dir()}?))"
            )
        concurrency = str(os.cpu_count() + 2)
        print_and_check_call([quantize_cmd, nonq_model_path, dest_type, concurrency])
    return q_model_path


def get_ggml_model_path(dirname: str, convert_type: str):
    if convert_type in ("0", "f32"):
        type_moniker = "f32"
    elif convert_type in ("1", "f16"):
        type_moniker = "f16"
    else:
        raise ValueError(f"Unknown type {convert_type}")
    model_path = os.path.join(dirname, f"ggml-model-{type_moniker}{GG_MODEL_EXTENSION}")
    return model_path


def convert_pth(
    dirname,
    *,
    convert_type: str,
    vocab_type: str,
    use_convert_hf_to_gguf=False,
):
    model_path = get_ggml_model_path(dirname, convert_type)
    try:
        stat = os.stat(model_path)
        if stat.st_size < 65536:
            print(f"Not believing a {stat.st_size:d}-byte model is valid, reconverting")
            raise FileNotFoundError()
    except FileNotFoundError:
        if use_convert_hf_to_gguf:
            convert_using_hf_to_gguf(dirname, convert_type=convert_type)
        else:
            convert_using_convert(
                dirname, convert_type=convert_type, vocab_type=vocab_type
            )
    return model_path


def convert_using_convert(dirname, *, convert_type, vocab_type):
    convert_hf_to_gguf_py = os.path.join(get_llama_cpp_dir(), "convert.py")
    if not os.path.isfile(convert_hf_to_gguf_py):
        raise RuntimeError(
            f"Could not find convert.py at {convert_hf_to_gguf_py} "
            f"(set LLAMA_CPP_DIR (currently {get_llama_cpp_dir()}?))"
        )
    print_and_check_call(
        [
            PYTHON_EXE,
            convert_hf_to_gguf_py,
            dirname,
            f"--outtype={convert_type}",
            f"--vocab-type={vocab_type}",
        ]
    )


def convert_using_hf_to_gguf(dirname, *, convert_type):
    convert_hf_to_gguf_py = os.path.join(get_llama_cpp_dir(), "convert-hf-to-gguf.py")
    if not os.path.isfile(convert_hf_to_gguf_py):
        raise RuntimeError(
            f"Could not find convert.py at {convert_hf_to_gguf_py} "
            f"(set LLAMA_CPP_DIR (currently {get_llama_cpp_dir()}?))"
        )
    print_and_check_call(
        [
            PYTHON_EXE,
            convert_hf_to_gguf_py,
            dirname,
            f"--outtype={convert_type}",
            "--verbose",
        ]
    )


def convert_pth_to_types(
    dirname,
    *,
    types,
    remove_nonquantized_model=False,
    nonquantized_type: str,
    vocab_type: str,
    use_convert_hf_to_gguf=False,
):
    # If f32 is requested, or a quantized type is requested, convert to fp32 GGML
    nonquantized_path = None
    if nonquantized_type in types or any(t.startswith("q") for t in types):
        nonquantized_path = convert_pth(
            dirname,
            convert_type=nonquantized_type,
            vocab_type=vocab_type,
            use_convert_hf_to_gguf=use_convert_hf_to_gguf,
        )
    # Other types
    for type in types:
        if type.startswith("q"):
            q_model_path = quantize(
                dirname,
                src_type=nonquantized_type,
                dest_type=type.upper(),
            )
            yield q_model_path
        elif type in ("f16", "f32") and type != nonquantized_type:
            yield convert_pth(dirname, convert_type=type)
        elif type == nonquantized_type:
            pass  # already dealt with
        else:
            raise ValueError(f"Unknown type {type}")
    if nonquantized_type not in types and remove_nonquantized_model:
        nonq_model_path = get_ggml_model_path(dirname, nonquantized_type)
        print(f"Removing non-quantized model {nonq_model_path}")
        os.remove(nonq_model_path)
    elif nonquantized_path:
        yield nonquantized_path


def download_repo(repo, dirname):
    files = list(huggingface_hub.list_repo_tree(repo, token=hf_token))
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
        "--nonquantized-type",
        type=str,
        choices=("f16", "f32"),
        default="f32",
        help="Dtype of the non-quantized model (default: %(default)s)",
    )
    ap.add_argument(
        "--keep-nonquantized",
        action="store_true",
        help="Don't remove the nonquantized model after quantization (unless it's explicitly requested)",
    )
    ap.add_argument(
        "--vocab-type",
        type=str,
        default="spm",
    )
    ap.add_argument(
        "--use-convert-hf-to-gguf",
        action="store_true",
        help="Use convert_hf_to_gguf.py instead of convert.py",
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
            remove_nonquantized_model=not args.keep_nonquantized,
            nonquantized_type=args.nonquantized_type,
            vocab_type=args.vocab_type,
            use_convert_hf_to_gguf=args.use_convert_hf_to_gguf,
        )
    )
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
