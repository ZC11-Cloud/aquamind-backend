import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "metrics.yaml"
RESULTS_DIR = ROOT_DIR / "results"


def load_metrics_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw_text": response.text}


def register_test_user(base_url: str, account: dict[str, Any], timeout: int = 20) -> None:
    payload = {
        "username": account["username"],
        "password": account["password"],
        "real_name": account.get("real_name"),
        "phone": account.get("phone"),
        "email": account.get("email"),
    }
    resp = requests.post(f"{base_url}/user/register", json=payload, timeout=timeout)
    # 200: 注册成功；400: 已存在，均视为可继续
    if resp.status_code not in (200, 400):
        raise RuntimeError(f"注册测试账号失败: status={resp.status_code}, body={resp.text}")


def login_and_get_token(base_url: str, account: dict[str, Any], timeout: int = 20) -> str:
    data = {"username": account["username"], "password": account["password"]}
    resp = requests.post(f"{base_url}/user/login", data=data, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"登录失败: status={resp.status_code}, body={resp.text}")
    payload = safe_json(resp)
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("登录返回中缺少 access_token")
    return token


def auth_header(token: str | None, auth_mode: str = "normal") -> dict[str, str]:
    if auth_mode == "none":
        return {}
    if auth_mode == "invalid":
        return {"Authorization": "Bearer invalid.token.value"}
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def timed_request(
    method: str,
    url: str,
    timeout: int,
    headers: dict[str, str] | None = None,
    json_payload: dict[str, Any] | None = None,
    raw_body: str | None = None,
) -> tuple[requests.Response, float]:
    begin = time.perf_counter()
    kwargs: dict[str, Any] = {"timeout": timeout, "headers": headers or {}}
    if json_payload is not None:
        kwargs["json"] = json_payload
    if raw_body is not None:
        kwargs["data"] = raw_body
        kwargs["headers"] = {
            **(headers or {}),
            "Content-Type": "application/json",
        }
    resp = requests.request(method=method.upper(), url=url, **kwargs)
    latency_ms = (time.perf_counter() - begin) * 1000
    return resp, latency_ms


def build_results_path(category: str, filename_prefix: str) -> Path:
    ts = now_str()
    return RESULTS_DIR / category / f"{filename_prefix}_{ts}.json"
