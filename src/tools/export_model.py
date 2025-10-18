"""
Export a slimmed PyTorch model to TorchScript (.pt) or ONNX (.onnx) format.

This script loads the slimmed (pruned) model, ensures it is cleaned from
pruning masks, and exports it in one of two supported formats for deployment.

Usage
-----
From project root, run one of:

    python tools/export_model.py --format torchscript
    python tools/export_model.py --format onnx

Requirements
------------
- Must run after pruning/export_slimmed_model has been called post-training.
- Uses config.yaml to locate checkpoint path and model definition.

Author: [Tu nombre o iniciales]
"""

import os
import argparse
import torch
import torch.nn.utils.prune as prune
from model.cnn3d import ViolenceDualStreamNet
from utils.config_loader import ConfigLoader


def load_config():
    """
    Load YAML configuration.

    Returns
    -------
    config : object
        Loaded configuration object.
    """
    config_path = os.path.normpath(os.path.join("config", "config.yaml"))
    return ConfigLoader(config_path).get()


def load_slim_model(config, device):
    """
    Load the slimmed pruned model (.pth) from checkpoint and clean it.

    Parameters
    ----------
    config : object
        Configuration object.

    device : torch.device
        Device on which to load the model.

    Returns
    -------
    model : torch.nn.Module
        Cleaned and ready-to-export PyTorch model.
    """
    model = ViolenceDualStreamNet(num_classes=config.model.num_classes)
    slim_path = config.output.checkpoints_path.replace(".pth", "_slim.pth")
    
    if not os.path.exists(slim_path):
        raise FileNotFoundError(f"Slim model not found: {slim_path}")

    state_dict = torch.load(slim_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Ensure pruning is removed if masks exist
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
            print(f"Removed pruning from: {name}")
    
    return model


def validate_equivalence(model, scripted_model, device):
    """
    Validate that the scripted/ONNX model behaves like the original one.

    Parameters
    ----------
    model : torch.nn.Module
        Original PyTorch model.

    scripted_model : torch.nn.Module
        Exported model to compare against.

    device : torch.device
        Device to run inference on.
    """
    model.eval()
    scripted_model.eval()

    dummy_input = torch.randn(1, 5, 16, 112, 112).to(device)
    with torch.no_grad():
        out1 = model(dummy_input)
        out2 = scripted_model(dummy_input)

    diff = torch.norm(out1 - out2).item()
    print(f"Output difference between original and exported: {diff:.6f}")
    assert diff < 1e-3, "Exported model behaves differently!"


def export_torchscript(model, path, device):
    """
    Export the model to TorchScript (.pt) format.

    Parameters
    ----------
    model : torch.nn.Module
        The model to export.

    path : str
        Destination file path.

    device : torch.device
        Computation device.
    """
    scripted = torch.jit.script(model)
    scripted.save(path)
    print(f"Exported TorchScript model to: {path}")
    validate_equivalence(model, scripted, device)


def export_onnx(model, path):
    """
    Export the model to ONNX (.onnx) format.

    Parameters
    ----------
    model : torch.nn.Module
        The model to export.

    path : str
        Destination file path.
    """
    dummy_input = torch.randn(1, 5, 16, 112, 112)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    print(f"Exported ONNX model to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", required=True, choices=["torchscript", "onnx"], help="Export format")
    args = parser.parse_args()

    config = load_config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = load_slim_model(config, device)
    export_dir = os.path.dirname(config.output.checkpoints_path)

    if args.format == "torchscript":
        export_path = os.path.join(export_dir, "model_scripted.pt")
        export_torchscript(model, export_path, device)

    elif args.format == "onnx":
        export_path = os.path.join(export_dir, "model.onnx")
        export_onnx(model, export_path)
