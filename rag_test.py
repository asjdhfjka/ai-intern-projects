
import os
# 设置 HuggingFace 镜像，防止下载 Embedding 模型时网络超时（国内必须）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
def build_vector_db():
    print("1. 正在读取 PDF...")
    # 加载你的 PDF 文件
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    documents = loader.load()

    print("2. 正在切分文本...")
    # 将长文本切分为每段 500 字，重叠部分 50 字，保持语义连贯
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   切分完成，共 {len(chunks)} 个文本片段")

    print("3. 正在加载 Embedding 模型（首次运行会下载约 100MB 模型，请耐心等待）...")
    # 使用一个轻量且支持中文的本地向量化模型
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    print("4. 正在存入向量数据库 Chroma...")
    # 将切分好的文本存入本地的 chroma_db 文件夹
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ 数据库构建完成！")
    return vector_db

if __name__ == "__main__":
    # 构建数据库
    vector_db = build_vector_db()

    print("\n--- 开始测试检索 ---")
    # 模拟提问，去数据库里找最相关的 3 段话
    query = "这本笔记讲了什么内容？"  # 你可以根据你的 PDF 内容改一下这句
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)

    print(f"关于问题：{query}")
    print(f"检索到 {len(results)} 个相关片段：\n")
    for i, doc in enumerate(results):
        print(f"【片段 {i+1}】: {doc.page_content[:200]}...\n")