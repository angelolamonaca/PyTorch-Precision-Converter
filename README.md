# PyTorch Precision Converter

![Badge](https://img.shields.io/badge/PyTorch-Converter-orange) ![Version](https://img.shields.io/badge/version-1.0.0-blue)

## Overview

`PyTorch Precision Converter` is a robust utility tool designed to convert the tensor precision of PyTorch model checkpoints and `safetensors` files. With the increasing need for efficient model deployment on various platforms, especially where memory or computational efficiency is paramount, converting models to reduced precision formats like `fp16` or `bf16` can be immensely beneficial. This tool provides the flexibility to convert not just traditional PyTorch checkpoints but also models saved in the custom `safetensors` format.

Key features include:
- Conversion of models to different precision formats.
- Option to select between different model configurations, like full model, only the Exponential Moving Average (EMA) parameters, or excluding the EMA parameters.
- Saving capability in both the custom `safetensors` format and the usual PyTorch checkpoint format.

## Features

- **Multiple Source Formats**: Load models from both PyTorch checkpoints and `safetensors` files.
- **Precision Conversion**: Convert tensors to different precision formats: `fp32`, `fp16`, and `bf16`.
- **Model Type Conversion**: Choose to convert:
    - The full model
    - Only the EMA parameters
    - Exclude the EMA parameters
- **Format Flexibility**: Save converted models in either the `safetensors` format or the standard PyTorch checkpoint format.

## Requirements

- Python 3.7 or higher
- PyTorch 1.7.0 or higher
- `safetensors` library

## Installation

1. Clone the GitHub repository:
    ```bash
    git clone https://github.com/angelolamonaca/PyTorch-Precision-Converter.git
    cd PyTorch-Precision-Converter
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Use the converter through the command line:

```bash
python converter.py -f <path_to_model> -p <precision> -t <conversion_type> -st
```

- `-f` or `--file`: Path to the model (PyTorch checkpoint or `safetensors` file). Default is `model.ckpt`.
- `-p` or `--precision`: Desired tensor precision (`fp32`, `fp16`, or `bf16`). Default is `fp32`.
- `-t` or `--type`: Conversion type (`full`, `ema-only`, `no-ema`). Default is `full`.
- `-st` or `--safe-tensors`: Flag to save the model in `safetensors` format. By default, it saves in PyTorch format.

## Examples

Convert a model saved in `safetensors` format to half precision (`fp16`):

```bash
python converter.py -f my_model.safetensors -p fp16
```

Convert only the EMA parameters of a PyTorch checkpoint to `bf16` and save in `safetensors` format:

```bash
python converter.py -f my_model.ckpt -p bf16 -t ema-only -st
```

## Contributing

We welcome contributions to the `PyTorch Precision Converter`. If you wish to contribute, kindly follow the standard GitHub pull request process:

1. Fork the repository.
2. Clone, create a new branch, make changes, and push them to your fork.
3. Open a pull request.

Ensure that your code adheres to the style and conventions of the existing codebase.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the PyTorch team and the OpenAI community for their continuous contributions to the machine learning ecosystem.
