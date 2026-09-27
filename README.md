# 工大智政 - 高校行政智能问答与审查系统

基于 RAG 技术的高校行政知识问答系统，支持智能问答和材料审查。

## 核心功能
- 🔍 智能问答：基于本地知识库的精准问答 + 多轮对话 + 溯源
- 📋 材料审查：自动抽取规则 + 场景隔离 + 结构化评分
- 📄 多格式支持：PDF / Word / TXT / 图片 / 扫描件 OCR

## 技术栈
- 后端：FastAPI + LangChain + Chroma
- 大模型：火山引擎方舟（DeepSeek-V4）
- 嵌入模型：BAAI/bge-small-zh-v1.5
- Rerank：BAAI/bge-reranker-base
- OCR：RapidOCR + PyMuPDF

## 运行方式
\`\`\`bash
pip install -r requirements.txt
python rag_test.py         # 构建知识库
uvicorn main:app --reload  # 启动服务
\`\`\`

访问 http://127.0.0.1:8000/
