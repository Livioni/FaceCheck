import os
import zipfile
from pathlib import Path

import requests


def request_facecheck_pipeline(
    server_base_url: str,
    image_path: str,
    save_zip_path: str = "facecheck_outputs.zip",
    extract_dir: str = "facecheck_outputs",
    timeout_sec: int = 300,
) -> list[str]:
    url = server_base_url.rstrip("/") + "/pipeline"

    image_path = str(Path(image_path).expanduser().resolve())
    save_zip_path = str(Path(save_zip_path).expanduser().resolve())
    extract_dir = str(Path(extract_dir).expanduser().resolve())

    with open(image_path, "rb") as f:
        files = {
            "image": (os.path.basename(image_path), f, "application/octet-stream"),
        }
        resp = requests.post(url, files=files, timeout=timeout_sec)

    resp.raise_for_status()

    with open(save_zip_path, "wb") as out:
        out.write(resp.content)

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
    # 把这里改成你那台跑服务的机器的内网 IP + 端口
    SERVER = "http://10.79.182.49:8000"
    IMAGE = "IMG_1889.PNG"

    files = request_facecheck_pipeline(
        server_base_url=SERVER,
        image_path=IMAGE,
        save_zip_path="outputs.zip",
        extract_dir="outputs_unzipped",
    )

    print("解压后文件列表：")
    for p in files:
        print(" -", p)