import os
import unicodedata


def normalize_unicode_nfc(path: str) -> str:
    return unicodedata.normalize("NFC", path)


def fix_unicode_filenames(root_dir):
    for current_root, dirs, files in os.walk(root_dir, topdown=False):
        # Normalize files
        for name in files:
            original_path = os.path.join(current_root, name)
            normalized_name = normalize_unicode_nfc(name)
            normalized_path = os.path.join(current_root, normalized_name)

            if normalized_name != name:
                print(f"Renaming file:\n  {original_path}\n→ {normalized_path}")
                os.rename(original_path, normalized_path)

        # Normalize directories (must be done after files, and from bottom up!)
        for name in dirs:
            original_path = os.path.join(current_root, name)
            normalized_name = normalize_unicode_nfc(name)
            normalized_path = os.path.join(current_root, normalized_name)

            if normalized_name != name:
                print(f"Renaming directory:\n  {original_path}\n→ {normalized_path}")
                os.rename(original_path, normalized_path)


# 🔧 Set the path to your dataset root folder
dataset_root = "/workspace/kaz-mm/benchmark/valid_dataset_kazvision_26042025"
fix_unicode_filenames(dataset_root)
