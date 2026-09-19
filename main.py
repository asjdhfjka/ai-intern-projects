
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# 1. 读取 .env 文件里的密钥
load_dotenv()

app = FastAPI()

# 2. 初始化火山引擎客户端
client = AsyncOpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)
# 加载本地向量数据库
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 防止下载超时
print("正在加载向量数据库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
print("向量数据库加载完成！")

@app.post("/chat")
async def chat(user_input: str):
    # 3. 调用大模型
    response = await client.chat.completions.create(
        # ⚠️ 注意！这里必须填你在火山引擎控制台创建的推理接入点 ID（ep-xxx 开头）
        # 千万不要填 doubao-pro 之类的模型名字！
        model="ep-20260919155615-h2mfv",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
    return {"reply": response.choices[0].message.content}
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(user_input: str):
    async def event_generator():
        # 1. 去向量数据库检索相关文档
        docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. 构建带有上下文提示的 messages
        system_prompt = f"""你是一个知识库问答助手。请严格根据以下提供的参考资料回答用户的问题。
如果资料中没有相关信息，请直接说“根据现有资料，我无法回答这个问题”，不要自己编造。

参考资料：
{context}
"""
        # 调用大模型
        response = await client.chat.completions.create(
            model="ep-20260919155615-h2mfv",  # ⚠️ 保留你自己的 ep-xxx
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            stream=True
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(event_generator(), media_type="text/event-stream")
# 挂载静态文件夹，让 / 路径直接显示 index.html
app.mount("/", StaticFiles(directory="static", html=True), name="static")