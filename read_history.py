import json
import pathlib


def print_user_messages(file_path: str):
    path = pathlib.Path(file_path)
    if not path.exists():
        print(f"File '{file_path}' not found.")
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")
        return

    print(f"=== {file_path} ===")
    for item in data.get("items", []):
        if item.get("role") == "user":
            for content in item.get("content", []):
                text = content.get("text")
                if text is not None:
                    first_line = text.replace("\r", " ").replace("\n", " ")
                    if len(first_line) > 120:
                        first_line = first_line[:117] + "..."
                    print("-", first_line)


if __name__ == "__main__":
    for fname in ["history.json", "history2.json"]:
        print_user_messages(fname)