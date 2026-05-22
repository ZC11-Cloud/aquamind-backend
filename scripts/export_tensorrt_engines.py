"""Export local TensorRT engine files for the bundled YOLO models.

TensorRT .engine files are machine-specific. Run this script on the
deployment computer so the generated engines match its GPU, CUDA, driver,
and TensorRT runtime.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MODEL_NAMES = ("yolo11", "yolov8", "yolo26")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def configure_ultralytics_config_dir(backend_dir: Path) -> None:
    config_dir = backend_dir / ".cache" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    version_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    local_packages = backend_dir / ".cache" / f"python-packages-{version_tag}"
    if local_packages.exists():
        local_packages_str = str(local_packages)
        sys.path[:] = [path for path in sys.path if path != local_packages_str]
        sys.path.insert(0, local_packages_str)


def export_with_trtexec(model_dir: Path, output_name: str) -> Path:
    onnx_path = model_dir / "best.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX source: {onnx_path}")

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("trtexec was not found on PATH")

    output_path = model_dir / output_name
    run([trtexec, f"--onnx={onnx_path}", f"--saveEngine={output_path}"])
    return output_path


def export_with_ultralytics(model_dir: Path, output_name: str, device: str) -> Path:
    pt_path = model_dir / "best.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing PyTorch source: {pt_path}")

    with tempfile.TemporaryDirectory(prefix=f"{model_dir.parent.parent.name}_trt_") as tmp:
        tmp_model = Path(tmp) / output_name.replace(".engine", ".pt")
        shutil.copy2(pt_path, tmp_model)
        from ultralytics import YOLO

        exported = YOLO(str(tmp_model)).export(format="engine", device=device)
        tmp_engine = Path(exported) if exported else tmp_model.with_suffix(".engine")
        if not tmp_engine.exists():
            raise FileNotFoundError(f"Ultralytics did not create {tmp_engine}")
        output_path = model_dir / output_name
        shutil.copy2(tmp_engine, output_path)
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate best_local.engine files on this deployment machine."
    )
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Backend project directory. Defaults to this script's parent backend directory.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="Model folders to export.",
    )
    parser.add_argument(
        "--source",
        choices=("pt", "onnx"),
        default="pt",
        help="Use .pt via Ultralytics export or .onnx via trtexec.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device for Ultralytics export when --source pt is used.",
    )
    parser.add_argument(
        "--output-name",
        default="best_local.engine",
        help="Engine filename written inside each model's weights directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_ultralytics_config_dir(args.backend_dir)
    weights_root = args.backend_dir / "weights"
    exported: list[Path] = []

    for model_name in args.models:
        model_dir = weights_root / model_name / "weights"
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing model directory: {model_dir}")

        if args.source == "pt":
            exported.append(export_with_ultralytics(model_dir, args.output_name, args.device))
        else:
            exported.append(export_with_trtexec(model_dir, args.output_name))

    print("\nExported local TensorRT engines:")
    for path in exported:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
