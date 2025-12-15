import os
import json

def make_item_for_dir(folder, root):
    items = []
    # 先收集本目录下的 .md 文件
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and name.lower().endswith(".md"):
            items.append({
                "label": os.path.splitext(name)[0],
                "file": os.path.relpath(full, root)  # 从 root 开始的相对路径
            })
    # 再收集子目录，递归生成
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isdir(full):
            sub = make_item_for_dir(full, root)
            if sub:  # 仅在子目录内有内容时加入
                items.append({
                    "label": name,
                    "items": sub
                })
    return items

def ensure_root_files(root):
    dir_name = os.path.basename(root)
    # 确保 config.json 存在
    config_path = os.path.join(root, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"import": "config_zh", "name": dir_name}, f, ensure_ascii=False, indent=4)
        print(f"created: {config_path}")

    # 确保 README.md 存在（内容为当前目录名）
    readme_path = os.path.join(root, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {dir_name}\n")
        print(f"created: {readme_path}")

def create_new_teedoc_sub_dir(file_path=None):
    if file_path == None:
        root = os.path.dirname(__file__)
    else:
        root = file_path
    print(f"processing: {root}")
    ensure_root_files(root)
    data = {"items": make_item_for_dir(root, root)}  # 传入 root
    out_path = os.path.join(root, "sidebar.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"updated: {out_path}")

def _load_json_with_comments(path):
    # 仅移除不在字符串中的 // 注释
    with open(path, "r", encoding="utf-8") as f:
        cleaned_lines = []
        for raw in f:
            line = raw.rstrip("\n")
            in_string = False
            escape = False
            i = 0
            out = []
            while i < len(line):
                ch = line[i]
                if escape:
                    out.append(ch)
                    escape = False
                else:
                    if ch == "\\":
                        out.append(ch)
                        escape = True
                    elif ch == '"':
                        in_string = not in_string
                        out.append(ch)
                    elif not in_string and ch == "/" and i + 1 < len(line) and line[i+1] == "/":
                        # 注释开始，丢弃本行后续
                        break
                    else:
                        out.append(ch)
                i += 1
            cleaned_lines.append("".join(out))
    return json.loads("\n".join(cleaned_lines))

def _dump_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def update_site_config_routes(doc_dir, site_config_path):
    if not os.path.isdir(doc_dir):
        print(f"doc dir not found: {doc_dir}")
        return
    if not os.path.isfile(site_config_path):
        print(f"site_config not found: {site_config_path}")
        return

    site_cfg = _load_json_with_comments(site_config_path)
    route = site_cfg.setdefault("route", {})
    docs = route.setdefault("docs", {})

    # 收集 doc 目录下的一级子目录
    for name in sorted(os.listdir(doc_dir)):
        full = os.path.join(doc_dir, name)
        if os.path.isdir(full):
            key = f"/{name}/"
            val = f"doc/{name}"
            if docs.get(key) != val:
                docs[key] = val
                print(f"route.docs added/updated: {key} -> {val}")

    _dump_json(site_config_path, site_cfg)
    print(f"updated: {site_config_path}")


def update_locale_nav_items(doc_dir, config_template_dir):
    # 目标：更新 config/config_zh.json 的 navbar.items
    cfg_path = os.path.join(config_template_dir, "config_zh.json")
    if not os.path.isfile(cfg_path):
        print(f"locale config not found: {cfg_path}")
        return
    cfg = _load_json_with_comments(cfg_path)

    navbar = cfg.setdefault("navbar", {})
    items = navbar.setdefault("items", [])

    # 已有 URL 集合，避免重复
    existing_urls = set()
    for it in items:
        url = it.get("url")
        if isinstance(url, str):
            existing_urls.add(url)

    # 为 doc 下的一级目录添加菜单项
    added = 0
    for name in sorted(os.listdir(doc_dir)):
        full = os.path.join(doc_dir, name)
        if os.path.isdir(full):
            url = f"/{name}/"
            if url not in existing_urls:
                items.append({
                    "url": url,
                    "label": name,
                    "position": "left"
                })
                added += 1
                print(f"navbar.items added: {url}")

    if added:
        _dump_json(cfg_path, cfg)
        print(f"updated: {cfg_path}")
    else:
        print("no navbar items changes")

def main():
    doc_dir = os.path.dirname(__file__) + "\\doc"
    for name in os.listdir(doc_dir):
        full = os.path.join(doc_dir, name)
        if os.path.isdir(full):
            create_new_teedoc_sub_dir(full)
    site_config_path = os.path.join(os.path.dirname(__file__), "site_config.json")
    update_site_config_routes(doc_dir, site_config_path)

    # 新增：更新 locale(navbar.items)
    config_template_dir = os.path.join(os.path.dirname(__file__), "config")
    update_locale_nav_items(doc_dir, config_template_dir)

if __name__ == "__main__":
    main()