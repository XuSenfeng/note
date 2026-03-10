import os
import json

def make_item_for_dir(folder):
    items = []
    # 先收集本目录下的 .md 文件
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and name.lower().endswith(".md"):
            items.append({
                "label": os.path.splitext(name)[0],
                "file": name
            })
    # 再收集子目录，递归生成
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isdir(full):
            sub = make_item_for_dir(full)
            if sub:  # 仅在子目录内有内容时加入
                items.append({
                    "label": name,
                    "items": sub
                })
    return items

def main():
    root = os.path.dirname(__file__)
    data = {"items": make_item_for_dir(root)}
    out_path = os.path.join(root, "sidebar.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"updated: {out_path}")

if __name__ == "__main__":
    main()