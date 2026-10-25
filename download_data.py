import os
import requests
import zipfile
from config import RAW_DIR

REPO_URL = "https://github.com/irsafilo/KION_DATASET"
RAW_BASE = "https://raw.githubusercontent.com/irsafilo/KION_DATASET/main/"

FILES_TO_DOWNLOAD = [
    "interactions.csv",
    "users.csv",
    "items.csv",
]


def download_file(url, save_path):
    print(f"Скачиваю {url} ...")
    response = requests.get(url, stream=True, timeout=120)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Сохранено: {save_path}")
        return True
    else:
        print(f"  Ошибка {response.status_code} при скачивании {url}")
        return False


def try_download_from_raw(filename):
    url = RAW_BASE + filename
    save_path = os.path.join(RAW_DIR, filename)
    return download_file(url, save_path)


def try_download_from_data_folder(filename):
    url = RAW_BASE + "data/" + filename
    save_path = os.path.join(RAW_DIR, filename)
    return download_file(url, save_path)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    all_ok = True
    for filename in FILES_TO_DOWNLOAD:
        save_path = os.path.join(RAW_DIR, filename)
        if os.path.exists(save_path):
            print(f"Файл {filename} уже существует, пропускаю")
            continue

        success = try_download_from_raw(filename)
        if not success:
            success = try_download_from_data_folder(filename)
        if not success:
            all_ok = False

    if not all_ok:
        print()
        print("=" * 60)
        print("Не все файлы удалось скачать автоматически.")
        print(f"Скачайте данные вручную с {REPO_URL}")
        print(f"и поместите файлы interactions.csv, users.csv, items.csv в:")
        print(f"  {RAW_DIR}")
        print("=" * 60)
    else:
        print()
        print("Все файлы успешно скачаны!")


if __name__ == "__main__":
    main()
