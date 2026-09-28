import os
import zipfile
import hashlib
# 设置 HuggingFace 镜像，防止下载 Embedding 模型时网络超时（国内必须）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import fitz  # PyMuPDF
from langchain_core.documents import Document

# ⭐ 保留 RapidOCR
from rapidocr_onnxruntime import RapidOCR
ocr_engine = RapidOCR()


def extract_text_from_image(image_path: str) -> str:
    """用 RapidOCR 识别图片里的文字"""
    try:
        result, _ = ocr_engine(image_path)
        if result:
            return "\n".join([line[1] for line in result])
        return ""
    except Exception as e:
        print(f"❌ OCR 识别失败: {e}")
        return ""


def ocr_pdf_if_empty(pdf_path: str, documents) -> list:
    """如果是扫描版 PDF（提取不出字），则自动转图片并 OCR"""
    total_chars = sum([len(doc.page_content) for doc in documents])
    if total_chars > 50:
        return documents

    print(f"   ⚠️ 检测到扫描版 PDF，启动 RapidOCR 解析...")
    ocr_docs = []
    try:
        pdf_doc = fitz.open(pdf_path)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            pix = page.get_pixmap(dpi=150)
            temp_img_path = f"./temp_ocr_page_{page_num}.png"
            pix.save(temp_img_path)

            page_text = extract_text_from_image(temp_img_path)

            if page_text.strip():
                ocr_docs.append(Document(
                    page_content=page_text,
                    metadata={"source": pdf_path, "page": page_num + 1}
                ))

            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        print(f"   ✅ 扫描版 PDF OCR 完成，共解析 {len(ocr_docs)} 页")
        return ocr_docs
    except Exception as e:
        print(f"❌ PDF OCR 失败: {e}")
        return documents


def smart_docx_loader(docx_path: str, filename: str):
    """Word 图文混排智能解析：解压 zip 提取图片并 OCR"""
    try:
        loader = Docx2txtLoader(docx_path)
        text_docs = loader.load()
        base_text = text_docs[0].page_content if text_docs else ""
    except Exception as e:
        print(f"⚠️ Word 文本提取失败: {e}")
        base_text = ""

    image_texts = []
    try:
        with zipfile.ZipFile(docx_path, 'r') as z:
            for name in z.namelist():
                if name.startswith('word/media/') and name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_bytes = z.read(name)
                    temp_img = f"./temp_docx_{name.split('/')[-1]}"
                    with open(temp_img, "wb") as f:
                        f.write(img_bytes)

                    ocr_text = extract_text_from_image(temp_img)
                    if ocr_text.strip():
                        image_texts.append(ocr_text)

                    if os.path.exists(temp_img):
                        os.remove(temp_img)
    except Exception as e:
        print(f"⚠️ Word 图片提取失败: {e}")

    combined_text = base_text
    if image_texts:
        combined_text += "\n\n【Word内嵌图片识别结果】：\n" + "\n".join(image_texts)
        print(f"   ✅ Word 内嵌图片 OCR 完成，识别出 {len(image_texts)} 张图片内容")

    if combined_text.strip():
        return [Document(page_content=combined_text, metadata={"source": filename, "page": 1})]
    return []


def build_vector_db():
    print("1. 正在读取 data 文件夹下的所有通知文件...")

    if not os.path.exists("sources.json"):
        print("❌ 找不到 sources.json，请先运行 auto_build_data.py 生成它！")
        return None

    with open("sources.json", "r", encoding="utf-8") as f:
        sources_map = json.load(f)

    data_dir = "./data"
    if not os.path.exists(data_dir):
        print("❌ 找不到 data 文件夹，请先运行 auto_build_data.py 抓取数据！")
        return None

    all_documents = []

    for filename in os.listdir(data_dir):
        if not filename.endswith((".pdf", ".txt", ".docx")):
            continue

        file_path = os.path.join(data_dir, filename)

        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                documents = ocr_pdf_if_empty(file_path, documents)
            elif filename.endswith(".docx"):
                documents = smart_docx_loader(file_path, filename)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
        except Exception as e:
            print(f"❌ 读取文件 {filename} 失败: {e}")
            continue

        if documents:
            total_chars = sum([len(doc.page_content) for doc in documents])
            print(f"📄 已读取文件: {filename}，总字数: {total_chars}")
        else:
            print(f"⚠️ 警告: {filename} 读取为空！")

        # 注入元数据
        meta = sources_map.get(filename, {})
        for doc in documents:
            doc.metadata["source_file"] = filename
            doc.metadata["source_url"] = meta.get("url", "")
            doc.metadata["source_title"] = meta.get("title", filename)
            doc.metadata["department"] = meta.get("department", "未知部门")
            doc.metadata["publish_date"] = meta.get("publish_date", "")
            if filename and not doc.page_content.startswith(f"【{filename}】"):
                doc.page_content = f"【{filename}】\n{doc.page_content}"

        all_documents.extend(documents)

    if not all_documents:
        print("❌ data 文件夹里没有找到任何有效文件。")
        return None

    # ============================================================
    # ⭐ 父子分块（Small-to-Big）
    # 父块：1500字，给大模型看完整上下文
    # 子块：400字，用于精准向量检索
    # ============================================================
    print(f"\n2. 正在切分文本（父子分块：父块1500字，子块400字）...")

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    chunks = []
    for doc in all_documents:
        source_file = doc.metadata.get("source_file", "未知文件")
        # 1. 先切父块
        parent_chunks = parent_splitter.split_documents([doc])
        for p_idx, parent_chunk in enumerate(parent_chunks):
            # ⭐ 修复：用 md5 生成稳定的父块 ID
            content_hash = hashlib.md5(parent_chunk.page_content.encode("utf-8")).hexdigest()[:8]
            parent_id = f"{source_file}__p{p_idx}__{content_hash}"

            # 2. 再对父块切子块
            child_chunks = child_splitter.split_documents([parent_chunk])
            for child_chunk in child_chunks:
                # ⭐ 关键：子块挂上父块信息
                child_chunk.metadata["parent_id"] = parent_id
                child_chunk.metadata["parent_content"] = parent_chunk.page_content
                chunks.append(child_chunk)
    # ⭐ 过滤掉过短的片段（比如纯标题）
    chunks = [c for c in chunks if len(c.page_content) > 50]
    print(f"   🧹 过滤短片段后剩余：{len(chunks)} 条")
    print(f"   切分完成（过滤前）：共 {len(chunks)} 个子块")

    # ⭐ 过滤掉过短的片段（在前缀拼接之前，纯内容长度 < 150 字）
    chunks = [c for c in chunks if len(c.page_content) > 150]
    print(f"   🧹 过滤短片段后：共 {len(chunks)} 个子块")

    # 给每个子块加来源前缀（方便向量检索命中）
    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "未知文件")
        chunk.page_content = f"【文档来源】：{source_file}\n{chunk.page_content}"

    print("3. 正在加载 Embedding 模型（首次运行会下载约 100MB 模型，请耐心等待）...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    print("4. 正在存入向量数据库 Chroma...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ 数据库构建完成！")
    return vector_db


if __name__ == "__main__":
    vector_db = build_vector_db()

    if vector_db:
        print("\n--- 开始测试检索 ---")
        query = "怎么交学费？"
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke(query)

        print(f"关于问题：{query}")
        print(f"检索到 {len(results)} 个相关片段：\n")
        for i, doc in enumerate(results):
            print(f"【片段 {i + 1}】")
            print(f"   来源：{doc.metadata.get('source_title', '未知')}")
            print(f"   网址：{doc.metadata.get('source_url', '无')}")
            print(f"   父块 ID：{doc.metadata.get('parent_id', '无')}")
            print(f"   子块内容：{doc.page_content[:150]}...")
            print(f"   父块内容预览：{doc.metadata.get('parent_content', '无')[:150]}...\n")