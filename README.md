# AI 实习冲刺项目集

大二学生，为了进入 AI 应用开发/大模型方向实习，从零构建的项目合集。

## 📦 项目一：全栈流式 AI 聊天机器人
- **技术栈**：Python, FastAPI, 火山引擎方舟 API, SSE (Server-Sent Events), 原生 HTML/CSS/JS
- **核心亮点**：
  - 搭建 FastAPI 异步后端，成功接入大模型（DeepSeek/豆包）。
  - 实现流式输出（Streaming），大幅降低首字响应延迟，实现打字机效果。
  - 编写原生前端页面，实现前后端分离。
- **效果展示**：
  *(把你之前那个聊天界面截图拖拽到这一段下面，面试官一眼就能看到)*

## 📚 项目二：基于 RAG 的动态知识库问答系统
- **技术栈**：LangChain, ChromaDB, HuggingFace Embedding, FastAPI
- **核心亮点**：
  1. 搭建 RAG 全链路：包含文档加载、文本切片、向量化与存储。
  2. 结合本地向量数据库与提示词工程，解决大模型幻觉问题。
  3. **支持动态知识库热更新**：开发前端上传接口与后端解析模块，支持用户上传 PDF/TXT 文件后，系统实时切分、向量化并增量入库。
  4. 结合 FastAPI 流式接口，实现基于新上传文档的精准流式问答。
- **效果展示**：
  *(把你刚才问“豆姐是谁”的截图拖拽到这里)*

## 🚀 本地运行指南
1. 克隆项目：`git clone https://github.com/asjdhfjka/ai-intern-projects.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 配置环境变量：新建 `.env` 文件，填入 `ARK_API_KEY=你的key`
4. 启动服务：`uvicorn main:app --reload`
