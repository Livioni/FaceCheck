import os
import zipfile
from pathlib import Path

import requests


def request_facecheck_heic(
    server_base_url: str,
    heic_path: str,
    save_zip_path: str = "heic_outputs.zip",
    extract_dir: str = "heic_outputs",
    timeout_sec: int = 600,
) -> list[str]:
    url = server_base_url.rstrip("/") + "/heic"

    heic_path = str(Path(heic_path).expanduser().resolve())
    save_zip_path = str(Path(save_zip_path).expanduser().resolve())
    extract_dir = str(Path(extract_dir).expanduser().resolve())

    with open(heic_path, "rb") as f:
        files = {
            "image": (os.path.basename(heic_path), f, "image/heic"),
        }
        resp = requests.post(url, files=files, timeout=timeout_sec)

    resp.raise_for_status()

    content = resp.content
    if not content.startswith(b"PK"):
        raise RuntimeError(
            f"服务端没有返回 zip（content-type={resp.headers.get('content-type', '?')}, "
            f"前200字节={content[:200]!r}）"
        )

    with open(save_zip_path, "wb") as out:
        out.write(content)

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(save_zip_path, "r") as zf:
        zf.extractall(extract_dir)

    all_files = []
    for root, _, files in os.walk(extract_dir):
        for name in files:
            all_files.append(os.path.join(root, name))
    all_files.sort()
    return all_files


if __name__ == "__main__":
    SERVER = "http://10.79.182.49:8000"
    HEIC = "IMG_0267.HEIC"

    files = request_facecheck_heic(
        server_base_url=SERVER,
        heic_path=HEIC,
        save_zip_path="heic.zip",
        extract_dir="heic_unzipped",
    )

    print("解压后文件列表：")
    for p in files:
        print(" -", p)
