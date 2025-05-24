import torch
import numpy as np
from scipy.io import loadmat, savemat
import argparse
import sys
import os

# ---- MATLAB / NumPy Interoperability ----

def load_mat_tensor(filename, key=None, to_torch=True, device='cpu'):
    """
    Load a variable from a MATLAB .mat file and return as numpy array or torch tensor.
    Args:
        filename (str): Path to .mat file.
        key (str): Variable name. If None, returns dict of all variables.
        to_torch (bool): If True, return as torch tensor.
        device (str): PyTorch device.
    Returns:
        tensor or dict of tensors/arrays
    """
    mat = loadmat(filename)
    # Remove MATLAB metadata keys
    mat = {k: v for k, v in mat.items() if not k.startswith('__')}
    if key:
        arr = mat[key]
        return torch.from_numpy(arr).to(device) if to_torch else arr
    else:
        if to_torch:
            return {k: torch.from_numpy(v).to(device) for k, v in mat.items()}
        return mat

def save_tensor_mat(filename, array_dict):
    """
    Save a dict of numpy arrays or torch tensors to a MATLAB .mat file.
    Args:
        filename (str): Output .mat file.
        array_dict (dict): {name: tensor/array}
    """
    save_dict = {}
    for k, v in array_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        save_dict[k] = v
    savemat(filename, save_dict)

def numpy_to_torch(arr, device='cpu'):
    """
    Convert a numpy array to a torch tensor.
    """
    return torch.from_numpy(arr).to(device)

def torch_to_numpy(tensor):
    """
    Convert a torch tensor to a numpy array.
    """
    return tensor.detach().cpu().numpy()

# ---- Command-Line Interface (CLI) ----

def run_cli():
    parser = argparse.ArgumentParser(
        description="Spectral-Spatial Pulse Designer CLI"
    )
    parser.add_argument('--config', type=str, help="YAML or JSON config file for pulse parameters")
    parser.add_argument('--output', type=str, help="Output file prefix (rf, grad, etc.)")
    parser.add_argument('--export', type=str, choices=['pta', 'mod', 'rf', 'txt'], help="Export format")
    parser.add_argument('--matlab', type=str, help="Save result as .mat file")
    parser.add_argument('--device', type=str, default='cpu', help="PyTorch device")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    args = parser.parse_args()

    # Dynamic import to avoid circular import
    from spectral_spatial_designer import SpectralSpatialPulseDesigner
    from export_tools import PulseExporter

    import yaml
    import json

    if not args.config:
        print("Error: --config must be provided (YAML or JSON file with pulse parameters)")
        sys.exit(1)
    # Load config
    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        with open(args.config, 'r') as f:
            params = yaml.safe_load(f)
    else:
        with open(args.config, 'r') as f:
            params = json.load(f)

    if args.verbose:
        print("Loaded configuration:")
        print(params)

    designer = SpectralSpatialPulseDesigner(device=args.device, **params.get('designer', {}))
    pulse = designer.design_pulse(**params['pulse'])

    rf = pulse['rf_G']
    grad = pulse['grad_G_cm']
    dt = designer.dt

    # Export if requested
    if args.export and args.output:
        exporter = PulseExporter(rf, grad, dt)
        outname = f"{args.output}.{args.export}"
        if args.export == 'pta':
            exporter.export_siemens_pta(outname)
        elif args.export == 'mod':
            exporter.export_ge_mod(outname)
        elif args.export == 'rf':
            exporter.export_philips_rf(outname)
        elif args.export == 'txt':
            exporter.export_pulseq_txt(outname)
        if args.verbose:
            print(f"Exported pulse to {outname}")

    # Save .mat if requested
    if args.matlab:
        save_tensor_mat(args.matlab, {'rf': rf, 'grad': grad, 'dt': np.array([dt])})
        if args.verbose:
            print(f"Saved .mat file: {args.matlab}")

if __name__ == '__main__':
    if sys.argv[0].endswith('interop_and_cli.py'):
        run_cli()
