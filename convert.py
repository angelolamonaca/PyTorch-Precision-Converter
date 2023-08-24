import argparse

import torch
from safetensors.torch import save_file, load_file
from torch import Tensor

# Initialize command-line argument parser
parser = argparse.ArgumentParser(description="Converts model tensor precision and saves to desired format.")
parser.add_argument("-f", "--file", type=str, default="model.ckpt", help="Path to the model file.")
parser.add_argument("-p", "--precision", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="Tensor precision: fp32 (default) / fp16 / bf16.")
parser.add_argument("-t", "--type", type=str, default="full", choices=["full", "ema-only", "no-ema"],
                    help="Model conversion type: full (default) / ema-only / no-ema.")
parser.add_argument("-st", "--safe-tensors", action="store_true", default=False,
                    help="Use safetensors model format for output.")

cmds = parser.parse_args()


# Functions to convert tensor precision
def conv_fp16(t: Tensor):
    return t.half() if isinstance(t, Tensor) else t


def conv_bf16(t: Tensor):
    return t.bfloat16() if isinstance(t, Tensor) else t


def conv_full(t):
    return t


# Dictionary to map user input to precision conversion functions
_g_precision_func = {
    "fp32": conv_full,
    "fp16": conv_fp16,
    "bf16": conv_bf16,
}


def convert(path: str, conv_type: str):
    """Convert model tensor precision based on user arguments."""
    converted_model = {}
    precision_func = _g_precision_func[cmds.precision]

    # Load the model from path
    if path.endswith(".safetensors"):
        model = load_file(path, device="cpu")
    else:
        model = torch.load(path, map_location="cpu")

    state_dict = model["state_dict"] if "state_dict" in model else model

    # Conversion based on type
    if conv_type == "ema-only":
        for k, v in state_dict.items():
            if k.startswith("model_ema"):
                converted_name = k.replace("model_ema.", "")
                converted_model[converted_name] = precision_func(v)
                print(f"ema: {k} > {converted_name}")
            elif k not in ["model_ema.num_updates", "model_ema.decay"]:
                converted_model[k] = precision_func(v)
                print(k)
    elif conv_type == "no-ema":
        for k, v in state_dict.items():
            if "model_ema" not in k:
                converted_model[k] = precision_func(v)
    else:
        for k, v in state_dict.items():
            converted_model[k] = precision_func(v)

    return converted_model


def main():
    """Main function to execute tensor conversion and saving."""
    # Extract model name from file path
    model_name = ".".join(cmds.file.split(".")[:-1])

    # Convert tensor precision
    converted = convert(cmds.file, cmds.type)

    # Save converted model
    save_name = f"{model_name}-{cmds.type}-{cmds.precision}"
    print("Conversion successful. Saving model...")

    if cmds.safe_tensors:
        save_file(converted, save_name + ".safetensors")
    else:
        torch.save({"state_dict": converted}, save_name + ".ckpt")

    print("Conversion and saving complete.")


if __name__ == "__main__":
    main()
