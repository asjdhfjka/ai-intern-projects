import os
import json
import trafilatura
import re
import time

# 1. 自动读取 urls.txt（不需要你改任何格式，直接把你之前复制的文本粘贴进去即可）
with open("urls.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

raw_urls = []
for i, line in enumerate(lines):
    if "http" in line:
        # 提取链接
        url_match = re.search(r'(https?://\S+)', line)
        if not url_match:
            continue
        url = url_match.group(1)

        # 提取标题（标题大概率在上一行，或者这一行的 "|" 前面）
        if "|" in line:
            pre_text = line.split("|")[0].strip()
            # 如果 "|" 前面是日期（如 2026-09-17），说明标题在上一行
            if re.match(r'\d{4}-\d{2}-\d{2}', pre_text):
                title = lines[i - 1]
            else:
                title = pre_text
        else:
            title = lines[i - 1] if i > 0 else "未知标题"

        raw_urls.append(f"{title} | {url}")

if not raw_urls:
    print("❌ 没能从 urls.txt 中解析出任何链接，请检查文件内容！")
    exit()

print(f"✅ 成功解析出 {len(raw_urls)} 条通知，开始自动提取正文...\n")

# 2. 配置数据保存路径
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
sources_map = {}

# 3. 遍历提取
for i, item in enumerate(raw_urls):
    try:
        title, url = item.split(" | ")
    except ValueError:
        continue

    print(f"[{i + 1}/{len(raw_urls)}] 正在提取：{title}")

    # 下载并提取正文
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print("   ❌ 网页下载失败")
        continue
    text = trafilatura.extract(downloaded)
    if not text:
        print("   ❌ 正文提取为空")
        continue

    import re
    text = text.replace("NBSP", " ").replace("&nbsp;", " ").replace("\xa0", " ")
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 去掉多余的空行

    # 清洗文件名
    safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)
    filename = f"{safe_title}.txt"
    file_path = os.path.join(DATA_DIR, filename)

    # 保存正文
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n\n{text}")

    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text[:200])  # 从正文开头找
    publish_date = date_match.group(1) if date_match else ""
    # 记录元数据
    sources_map[filename] = {
        "url": url,
        "title": title,
        "department": "教务处",
        "publish_date":publish_date
    }

    time.sleep(1)  # 礼貌爬取

# 4. 保存 sources.json
with open("sources.json", "w", encoding="utf-8") as f:
    json.dump(sources_map, f, ensure_ascii=False, indent=2)

print(f"\n🎉 提取完成！共成功 {len(sources_map)} 篇。")