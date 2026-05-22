import os
import site
import sys
from pathlib import Path


def _configure_local_runtime() -> None:
    backend_dir = Path(__file__).resolve().parent
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(backend_dir / ".cache" / "ultralytics")
    )

    user_site = site.getusersitepackages()
    sys.path[:] = [path for path in sys.path if path != user_site]

    version_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    local_packages = backend_dir / ".cache" / f"python-packages-{version_tag}"
    if local_packages.exists():
        local_packages_str = str(local_packages)
        sys.path[:] = [path for path in sys.path if path != local_packages_str]
        sys.path.insert(0, local_packages_str)

        # Preload the matched CUDA torch build before uvicorn or app imports.
        import torch  # noqa: F401


_configure_local_runtime()

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
