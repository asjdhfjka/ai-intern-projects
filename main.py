import json
import zipfile
import hashlib
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
import os
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
import hashlib
os.environ["NO_PROXY"] = "localhost,127.0.0.1,ark.cn-beijing.volces.com"
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from fastapi import UploadFile, File
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from fastapi.responses import FileResponse
current_date = datetime.now().strftime("%Y年%m月%d日")
ocr_engine = RapidOCR()

# ⭐ 领域模板（原 extract_rules.py 的内容，已迁移至此）
DOMAIN_TEMPLATES = {
    "奖学金评定": "奖学金申请资格、绩点要求、排名要求、材料清单、评审流程",
    "请假管理": "请假时长、审批层级、请假事由、佐证材料、销假要求",
    "学生干部": "职位分类、任职时长、表现要求、证明材料",
    "违纪处分": "违纪行为、处分等级、申诉流程、记录影响",
    "团员推优": "推优条件、团员资格、评议流程、票数要求",
    "综合素质测评": "体质测试、挂科情况、第二课堂时长、品德表现",
    "转专业申请": "转专业条件、成绩要求、申请流程",
    "宿舍管理": "入住条件、退宿流程、违纪处理",
    "助学贷款": "申请资格、家庭经济困难认定、还款方式",
}

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

def extract_text_from_image(image_path: str) -> str:
    """用 OCR 识别图片里的文字"""
    try:
        result, _ = ocr_engine(image_path)
        if result:
            # result 格式是 [[[坐标], "文字", 置信度], ...]
            text_lines = [line[1] for line in result]
            return "\n".join(text_lines)
        return ""
    except Exception as e:
        print(f"❌ OCR 识别失败: {e}")
        return ""


async def extract_rules_from_content(doc_content: str, doc_title: str, domain: str) -> list:
    """从规章制度文本中抽取结构化规则"""
    domain_desc = DOMAIN_TEMPLATES.get(domain, "通用审查规则")

    prompt = f"""你是高校行政规章制度分析专家。请从以下文档中抽取可执行的结构化审查规则。

    【文档标题】：{doc_title}
    【文档领域】：{domain}
    【文档正文】：
    {doc_content[:6000]}

    【规则 Schema（强制）】：
    每条规则必须包含以下 8 个字段，缺一不可：

    1. id：唯一标识，格式 "{domain}_R001"
    2. name：规则名（10-20字，说明约束什么）
    3. type：校验类型，只能是 required / regex / range / keyword / enum 之一
    4. field：被校验的字段名（如"学号"、"日期"、"签名"）
    5. severity：严重程度，必须是以下三档之一：
    - "critical"：致命问题（缺签名、缺盖章、票数不够、资格审查不过），扣 30 分
    - "error"：严重问题（格式错误、信息不全、逻辑不一致），扣 10 分
    - "warning"：建议问题（措辞不规范、排版不美观），扣 3 分
    6. message：未通过时的提示
    7. scene：⭐ 适用材料类型（数组，可多选）
    可选值：["票根", "选票", "申请表", "推荐表", "会议记录", "资格审查", "通知公告", "通用"]
    8. condition：（可选）规则适用的额外条件（如"仅大一学生"）

    【scene 判断指引（关键）】：
    请仔细阅读文档，判断每条规则约束的是哪种材料的哪个字段：
    - 提到"票根"、"选票"、"监票人"、"唱票" → scene: ["票根"] 或 ["选票"]
    - 提到"申请表"、"申请人"、"个人简历" → scene: ["申请表"]
    - 提到"资格审查"、"入团满一年"、"无挂科" → scene: ["资格审查"]
    - 提到"黑板"、"会议记录"、"发言时长" → scene: ["会议记录"]
    - 提到"通知"、"公告"、"时间安排" → scene: ["通知公告"]
    - 适用于所有材料的（如日期格式） → scene: ["通用"]

    【输出格式】（只输出合法 JSON 数组，不要任何解释）：
    [
        {{
            "id": "{domain}_R001",
            "name": "票根必须有监票人签名",
            "type": "keyword",
            "field": "正文",
            "keywords": ["监票人", "签名"],
            "severity": "error",
            "message": "票根缺少监票人签名",
            "scene": ["票根"]
        }},
        {{
            "id": "{domain}_R002",
            "name": "入团满一年方可推优",
            "type": "range",
            "field": "入团年限",
            "min": 1,
            "severity": "error",
            "message": "入团未满一年",
            "scene": ["资格审查"]
        }},
        {{
            "id": "{domain}_R003",
            "name": "日期格式必须为两位日期",
            "type": "regex",
            "field": "日期",
            "pattern": "^(0[1-9]|[12][0-9]|3[01])日$",
            "severity": "warning",
            "message": "日期格式应如09日",
            "scene": ["通用"]
        }}
    ]

    【最终要求】：
    - scene 必须是非空数组，绝对不能省略！
    - 如果一条规则适用于多种材料，写多个 scene。
    - 至少抽 5 条规则，最多 50 条，避免冗余。
    """
    try:
        res = await client.chat.completions.create(
            model="ep-20260919155615-h2mfv",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json\n", "", 1)
        return json.loads(raw)
    except Exception as e:
        print(f"⚠️ 规则抽取失败: {e}")
        return []

def ocr_pdf_if_empty(pdf_path: str, documents) -> list:
    """如果是扫描版 PDF（提取不出字），则自动转图片并 OCR"""
    # 1. 计算原本提取出的总字数
    total_chars = sum([len(doc.page_content) for doc in documents])
    if total_chars > 50:  # 如果已经有正常的文字，说明不是扫描件，直接返回
        return documents

    print("⚠️ 检测到扫描版 PDF，启动 OCR 解析...")
    ocr_docs = []
    try:
        # 2. 用 PyMuPDF 打开 PDF
        pdf_doc = fitz.open(pdf_path)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            # 3. 把每一页渲染成图片（150 DPI 清晰度足够）
            pix = page.get_pixmap(dpi=150)
            temp_img_path = f"./temp_ocr_page_{page_num}.png"
            pix.save(temp_img_path)

            # 4. 对这张图片进行 OCR
            page_text = extract_text_from_image(temp_img_path)

            # 5. 包装成 LangChain Document 对象
            from langchain_core.documents import Document
            if page_text.strip():
                ocr_docs.append(Document(
                    page_content=page_text,
                    metadata={"source": pdf_path, "page": page_num + 1}
                ))

            # 6. 清理临时图片
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        print(f"✅ 扫描版 PDF OCR 完成，共解析 {len(ocr_docs)} 页")
        return ocr_docs
    except Exception as e:
        print(f"❌ PDF OCR 失败: {e}")
        return documents
# 1. 读取 .env 文件里的密钥
load_dotenv()#就是一个写好的函数，去.env文件里面找我的api，把它放到操作系统的环境变量里面

app = FastAPI()#就是拿 FastAPI 框架，生产出一个属于你自己的 Web 应用实体（名叫 app）。等这个应用跑起来，
# FastAPI会自动送我一个交互式网页（/docs），方便和别人测试整个系统的接口

# 2. 初始化火山引擎客户端
client = AsyncOpenAI(
    api_key=os.getenv("ARK_API_KEY"),#就是在我的环境变量里面找到"ARK_API_KEY"
    base_url="https://ark.cn-beijing.volces.com/api/v3"#按量计费的火山引擎的网站
)
#创建一个“翻译官”，拿着你的钥匙（api_key），去火山引擎的“办事处”（base_url）调用大模型服务。SDK调用大模型的底层工具包
# 加载本地向量数据库
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  #写入HuggingFace，但用的是国内镜像网站
print("正在加载向量数据库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
#嵌入模型就是一个“翻译官”，这是北京智源研究院（BAAI）开源的一个专门针对中文优化的轻量级（small）向量模型。
# 它体积小、速度快，非常适合个人电脑或普通服务器跑本地知识库。
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
#向量数据库擅长语义模糊查找，告诉数据库把数据持久化保存在本地当前目录下的 chroma_db 文件夹里。这样下次重启程序，之前上传的知识库还在，不用重新导入。
# 告诉数据库，以后存文字或者查文字时，都用刚才加载的“翻译官”来转换。
# ========== 混合检索初始化（BM25 + 向量） ==========
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

print("正在构建 BM25 关键词索引...")
# 1. 从 Chroma 里把已入库的所有片段读出来，用于构建 BM25 内存索引
all_data = vector_db.get()
if all_data and all_data.get('documents'):
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data['documents'], all_data['metadatas'])
    ]
else:
    all_docs = []
print(f"✅ BM25 索引准备就绪，共 {len(all_docs)} 个片段")

# 2. BM25 关键词检索器（擅长“GXzhcx”、“教务处”等专有名词）
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 5

# 3. 向量语义检索器（擅长“怎么交学费”这类模糊语义）
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# 4. 手工实现混合检索（两路结果合并去重）
def hybrid_search(query: str, search_filter: dict = None):
    # ========== 阶段一：多路粗召回 ==========
    # 1. 向量检索
    if search_filter:
        vector_docs = vector_retriever.invoke(query, filter=search_filter)
    else:
        vector_docs = vector_retriever.invoke(query)

    # 2. BM25 检索
    bm25_docs = bm25_retriever.invoke(query)
    if search_filter:
        allowed = search_filter.get("topic")
        bm25_docs = [d for d in bm25_docs if d.metadata.get("topic") == allowed]

    # 3. 合并去重，保留 20 个候选给 Rerank
    combined = []
    seen = set()
    for doc in vector_docs + bm25_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    candidates = combined[:20]

    if not candidates:
        return []

    # ========== 阶段二：Rerank 精排 ==========
    try:
        # 构造 (query, document) 对
        pairs = [(query, doc.page_content) for doc in candidates]

        # Rerank 打分
        scores = reranker.predict(pairs)

        # 按分数降序排列
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        print(f"   🎯 Rerank 精排完成，最高分: {scored_docs[0][1]:.4f}，最低分: {scored_docs[-1][1]:.4f}")

        # ⭐ 动态阈值：取最高分的 30% 作为下限，同时不低于 0.5
        top_score = scored_docs[0][1]
        dynamic_threshold = max(0.1, top_score * 0.3)
        print(f"   📊 动态阈值: {dynamic_threshold:.2f}（最高分 {top_score:.2f} × 30%）")

        # 打印 Top 8 候选的分数分布（方便调试）
        for i, (doc, score) in enumerate(scored_docs[:8]):
            title = doc.metadata.get('source_title', doc.metadata.get('source_file', ''))[:30]
            print(f"      候选 {i + 1}: 分数 {score:.2f} | {title}")

        filtered = [(doc, score) for doc, score in scored_docs if score >= dynamic_threshold]

        if not filtered:
            filtered = scored_docs[:3]

        # ⭐ 父子分块核心：用父块内容替换子块，避免上下文断裂
        final_docs = []
        seen_parents = set()
        for doc, score in filtered[:5]:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                # 用父块的完整内容替换子块的短内容
                new_doc = Document(
                    page_content=doc.metadata.get("parent_content", doc.page_content),
                    metadata=doc.metadata
                )
                final_docs.append(new_doc)
            elif not parent_id:
                # 兼容旧数据（没有父块信息的）
                final_docs.append(doc)

        return final_docs

    except Exception as e:
        print(f"⚠️ Rerank 失败，降级使用原始排序: {e}")
        return candidates[:5]

print("✅ 混合检索器（BM25 + 向量）准备就绪！")
from sentence_transformers import CrossEncoder

print("正在加载 Rerank 精排模型（首次运行会下载约 1GB，请耐心等待）...")
# BGE-Reranker-v2-m3 对中文支持极好，适合高校行政场景
reranker = CrossEncoder("BAAI/bge-reranker-base")
print("✅ Rerank 精排模型加载完成！")
#设置查询参数。意思是，每次用户提问，去数据库里找最相似的 6 个文本片段（k=6）。
print("向量数据库加载完成！")


def route_rules(matched_domain: str, matched_scene: str, all_rules: dict) -> list:
    """
    通用化规则匹配（三级优先级）：
    1. 领域 + scene 精准匹配 → 优先
    2. 领域 + 通用 scene → 次之
    3. 全局规则 → 兜底
    """
    applicable_rules = []

    is_specific_domain = (
            matched_domain
            and matched_domain not in ["通用审查", "其他", "未知", ""]
            and isinstance(all_rules.get("domains"), dict)
            and matched_domain in all_rules.get("domains", {})
    )

    if is_specific_domain:
        domain_rules = all_rules["domains"][matched_domain]["rules"]

        # 三级优先级
        exact_match = []  # scene 精准匹配
        general_match = []  # scene 包含"通用"
        no_scene = []  # 无 scene 字段（兼容旧数据）

        for r in domain_rules:
            rule_scenes = r.get("scene", [])
            # 兼容字符串格式
            if isinstance(rule_scenes, str):
                rule_scenes = [rule_scenes]
            # 兼容无 scene 的旧规则
            if not rule_scenes:
                no_scene.append(r)
                continue

            if matched_scene and matched_scene != "通用":
                if matched_scene in rule_scenes:
                    exact_match.append(r)
                elif "通用" in rule_scenes:
                    general_match.append(r)
            else:
                # 材料类型是"通用"，就取所有规则
                if "通用" in rule_scenes or len(rule_scenes) > 0:
                    general_match.append(r)

        # 如果精准匹配到，就只用精准匹配的（+通用）
        if exact_match:
            applicable_rules.extend(exact_match)
            applicable_rules.extend(general_match)
            print(f"   📌 领域【{matched_domain}】精准匹配 {len(exact_match)} 条 + 通用 {len(general_match)} 条")
        else:
            # 没有精准匹配，退而求其次：使用通用 + 无 scene 旧规则
            applicable_rules.extend(general_match)
            applicable_rules.extend(no_scene)
            print(
                f"   📌 领域【{matched_domain}】无精准匹配，使用通用 {len(general_match)} 条 + 兼容旧规则 {len(no_scene)} 条")
    else:
        print(f"   ⚠️ 材料类型为【{matched_domain}】，只使用全局规则")

    # 追加全局规则
    global_rules = all_rules.get("global_rules", [])
    applicable_rules.extend(global_rules)
    print(f"   📌 追加全局通用规则：{len(global_rules)} 条")

    # 按严重度排序
    severity_order = {"error": 0, "warning": 1, "info": 2}
    applicable_rules.sort(key=lambda r: (
        severity_order.get(r.get("severity", "warning"), 1),
        r.get("priority", 99)
    ))

    # 上限保护：最多 30 条
    if len(applicable_rules) > 30:
        print(f"   ⚠️ 规则数 {len(applicable_rules)} 条超限，只保留前 30 条")
        applicable_rules = applicable_rules[:30]

    return applicable_rules

def execute_hard_rules(facts: dict, rules: list) -> list:
    """对事实清单执行硬规则校验"""
    import re
    results = []

    for rule in rules:
        field = rule.get("field")
        value = facts.get(field, "")
        severity = rule.get("severity", "warning")
        passed = True

        rule_type = rule.get("type")

        if rule_type == "required":
            passed = bool(value and str(value).strip())
        elif rule_type == "regex":
            passed = bool(value and re.match(rule["pattern"], str(value)))
        elif rule_type == "keyword":
            content = facts.get("_full_text", "")
            passed = any(kw in content for kw in rule.get("keywords", []))
        elif rule_type == "range":
            try:
                num = float(value)
                passed = rule.get("min", 0) <= num <= rule.get("max", 999)
            except:
                passed = False
        elif rule_type == "enum":
            # 兼容 values / allowed 两种字段名
            allowed_values = rule.get("values") or rule.get("allowed") or []
            passed = str(value) in allowed_values
        results.append({
            "rule_id": rule["id"],
            "rule_name": rule["name"],
            "passed": passed,
            "severity": severity,
            "message": "" if passed else rule.get("message", "规则未通过")
        })

    return results

@app.post("/chat")
#大管家（app），如果外面有人用 POST 方式，敲门牌是 /chat 的这扇门，你就去执行下面这个叫 chat 的函数。
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
#拆包：从大模型返回的一坨复杂对象里，精准提取出回答文字：
# response.choices：大模型可能一次生成多个备选答案（choices），是个列表。
# [0]：取第一个（也是最常用的那个）。
# .message：拿到第一条消息对象。
# .content：拿到消息里的具体文本内容（比如“你好，很高兴见到你”）。
# 打包：把提取出的文本，包装成 Python 字典 {"reply": "模型的回答"}。
# FastAPI 会自动把这个字典转换成 JSON 格式返回给前端，前端收到后就是：
#chat 函数是你整个后端代码里，第一个真正干活的“打工人”。它的任务非常明确：接收用户提问，去问大模型，然后把大模型的回答拿回来。
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
# 和之前的 /chat 一样，这是告诉大管家（FastAPI），有人用 POST 敲门 /chat/stream，就执行这个函数
async def chat_stream(user_input: str, history: str = ""):
    async def event_generator():
        print(f"\n--- 收到请求: {user_input} ---")

        # 1. 查询重写（加回来，提升检索准确率）
        search_query = user_input
        if history:
            # 获取当前时间
            from datetime import datetime
            now_str = datetime.now().strftime("%Y年%m月")
            rewrite_prompt = f"""你是一个查询改写专家。请结合历史对话，将用户的最新问题改写成一个完整、独立、适合用于检索的问题。
            【当前时间】是：{now_str}。
            【重要规则】（请务必严格遵守）：
            1. ⭐【代词消解】如果用户的最新问题是一个简短的省略句（例如“啥职位啊”、“他的电话呢”、“那个怎么弄”），请**务必从历史对话中找到具体的实体（人名、文件名、事物名），补全主语**。
               （例如：历史对话在聊“何华儒表现如何”，用户问“啥职位啊”，你应该改写成“何华儒担任的职位是什么？”）
            2. 如果用户的最新问题是一个追问（包含“它”、“这个”、“那”等代词），请结合历史对话补全主语。
            3. ⭐【新话题隔离】如果用户的最新问题是一个完全无关的全新话题，请彻底忽略历史对话，绝对不要把旧话题的词汇混入新问题中。
               注意：如果用户的最新问题本身已经**包含了完整的语义**（如"那学校的学费怎么交"里已经明确了"学校的学费"），不要画蛇添足加"学生所在学校的"这种冗余前缀，直接用原意即可。
            4. 如果用户问的是“什么时候”、“时间”、“日期”，请在改写后的问题中明确加上“{now_str.split('年')[0]}年”或“最新”等时间限定词。
            5. 只输出改写后的问题，不要回答，不要解释。
            历史对话：
            {history}
            最新问题：{user_input}
            改写结果："""
            try:
                rewrite_res = await client.chat.completions.create(
                    model="ep-20260919155615-h2mfv",
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    temperature=0.1,
                    timeout=10.0
                )
                search_query = rewrite_res.choices[0].message.content.strip()
                if "改写结果：" in search_query:
                    search_query = search_query.split("改写结果：")[-1].strip()
                print(f"🔄 查询重写: {user_input} -> {search_query}")
            except Exception as e:
                print(f"⚠️ 查询重写失败，使用原问题: {e}")

        print(f"⏳ 开始本地向量检索，检索词: {search_query}")
        try:
            docs = await asyncio.to_thread(hybrid_search, search_query)
            print(f"✅ 检索完成，找到 {len(docs)} 个片段")
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            docs = []

        # ========== CRAG 逐 chunk 相关性过滤（并行版） ==========
        print("⏳ CRAG 逐 chunk 相关性过滤（并行）...")

        async def judge_one(doc):
            """判断单条资料是否与问题相关"""
            judge_prompt = f"""判断以下资料是否与问题相关。只输出 YES 或 NO，不要解释。

        【问题】：{search_query}

        【资料】：
        {doc.page_content[:400]}

        只输出 YES 或 NO。"""
            try:
                judge_res = await client.chat.completions.create(
                    model="ep-20260919155615-h2mfv",
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,
                    timeout=3.0
                )
                verdict = judge_res.choices[0].message.content.strip().upper()
                return "YES" in verdict
            except Exception:
                return True  # 超时或失败默认保留

        # 只对前 5 条并行判断
        candidates_to_judge = docs[:5]
        verdicts = await asyncio.gather(*[judge_one(d) for d in candidates_to_judge])
        relevant_docs = [doc for doc, ok in zip(candidates_to_judge, verdicts) if ok]
        filtered_count = len(candidates_to_judge) - len(relevant_docs)

        print(f"   ✅ 并行过滤完成：LLM 判过 {len(relevant_docs)} 条，过滤掉 {filtered_count} 条")
        # ⭐ CRAG 后：用过滤后的 relevant_docs 重新生成 sources
        sources = []
        seen_urls = set()
        for doc in relevant_docs:
            url = doc.metadata.get("source_url", "")
            title = doc.metadata.get("source_title", "")
            file_name = doc.metadata.get("source_file", "")
            if not url and file_name:
                url = f"/preview/{file_name}"
                title = title or file_name
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({"index": len(sources) + 1, "title": title, "url": url})

        sources = sources[:5]
        url_to_index = {s["url"]: s["index"] for s in sources}
        # ========== 如果过滤后为空，走拒答逻辑 ==========
        if not relevant_docs:
            fallback_sources = []
            for doc in docs[:3]:
                url = doc.metadata.get("source_url", "")
                title = doc.metadata.get("source_title", "")
                file_name = doc.metadata.get("source_file", "")
                if not url and file_name:
                    url = f"/preview/{file_name}"
                    title = title or file_name
                if url:
                    fallback_sources.append({
                        "index": len(fallback_sources) + 1,
                        "title": title,
                        "url": url
                    })

            yield f"__SOURCES__{json.dumps(fallback_sources, ensure_ascii=False)}__END__\n"
            yield "根据现有资料，我无法准确回答这个问题。您可以参考下方我为您检索到的相关材料。"
            return

        # ⭐ 用过滤后的 relevant_docs 重新拼装 context
        context_parts = []
        for doc in relevant_docs:
            doc_url = doc.metadata.get("source_url", "")
            if not doc_url and doc.metadata.get("source_file"):
                doc_url = f"/preview/{doc.metadata.get('source_file')}"
            if doc_url in url_to_index:
                idx = url_to_index[doc_url]
                context_parts.append(f"【来源{idx}】\n{doc.page_content}")
            else:
                context_parts.append(doc.page_content)

        context = "\n\n".join(context_parts)
        print(f"   📦 最终 context 长度: {len(context)} 字")

        yield f"__SOURCES__{json.dumps(sources, ensure_ascii=False)}__END__\n"

        print("⏳ 开始调用大模型生成回答...")
        system_prompt = f"""你是一个高校行政问答助手。请严格根据以下提供的参考资料回答用户的问题。
        如果资料中完全没有提到用户询问的事项，请直接说“根据现有资料，我无法回答这个问题”，不要自己编造。

        【当前时间】：{current_date}
        【核心规则】（请务必严格遵守）：
        1. 参考资料已经过相关性精排，最相关的排在最前面。请优先关注排在前面的资料，后面的资料仅供参考。
        2. 如果用户问的是“四级缴费”，而排在前面的资料是关于“学费缴纳”的，请如实回答：“根据现有资料，未找到四级报名费的缴纳方式，资料中仅包含学费缴纳流程。建议咨询教务处。”
        3. 如果参考资料中出现了具体的日期、时间等细节，请直接提取并回答。
        4. 在回答时，请在相关句子的末尾标注 [来源X]（X是数字），绝对不要在正文里编写“来源说明”这类解释段落。
        5. 【兜底回答策略】：如果参考资料中没有直接写明用户问的内容，但**有部分相关的信息**（例如用户问“准备什么材料”，资料里写了“报名方式、学号、报名网址”），
        请不要直接拒绝回答。你可以说：“资料中未明确列出所需的具体材料清单，但根据报名通知，您可能需要提前准备学号、登录报名网站（https://cet-bm.neea.edu.cn/）
        并核对个人信息。建议您具体参考报名系统的提示。”**绝对不要凭空编造材料清单。**
        6. 【主语核对规则】：如果用户的问题是一个省略句（例如“啥职位啊”），而你拿到的参考资料里全是与上文无关的内容（比如上一轮聊的是某个人，这轮资料全是招聘会），
        请**不要强行用现有资料回答**。请直接说：“根据现有资料，未找到与【何华儒】相关的职位信息。”
        7. 【表格数据防串行规则】：如果参考资料是由表格 OCR 识别而来的扁平文本，包含多行多列数据，请**极其严格地核对每一行的内容**。
        绝对不允许将某一行的人名、职位、学号与其他行的内容进行交叉组合！如果发现姓名和职位/学号在不同行或无法100%确认对应关系，
        请如实回答“资料排版可能存在错位，无法准确确认”，不要强行拼凑。
        8. 【缴费场景强制区分】：
        - 如果用户问"四级/六级报名费"，只允许引用《全国大学英语四、六级考试报名》相关文档。
        - 如果用户问"学费/住宿费"，只允许引用《微信公众号缴费方式》《校园统一支付平台》相关文档。
        - 绝对不允许把"学费缴纳流程"当作"四级报名费缴纳方式"回答。
        参考资料：
        {context}
        """
        try:
            response = await client.chat.completions.create(
                model="ep-20260919155615-h2mfv",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                stream=True,
                timeout=20.0
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            print("✅ 回答完毕！")
        except Exception as e:
            print(f"❌ 大模型调用失败: {e}")
            yield "抱歉，网络请求超时或服务暂时不可用。"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
# 告诉 FastAPI，不要等水接满一桶再给前端，而是水龙头滴出一滴水，你就立刻通过网络把水传给前端。
# 挂载静态文件夹，让 / 路径直接显示 index.html
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # ========== 1. 安全防线：文件类型检查 ==========
    if not file.filename.endswith((".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg")):
        return {"error": "❌ 仅支持 PDF、TXT、Word、PNG、JPG 文件"}

    # ========== 2. 安全防线：文件大小限制（20MB） ==========
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > 20 * 1024 * 1024:
        return {"error": "❌ 文件过大，请上传 20MB 以内的文件"}

    # ========== 3. 防重复：计算文件 MD5（文件指纹） ==========
    file_content = await file.read()
    file_hash = hashlib.md5(file_content).hexdigest()
    await file.seek(0)

    existing_docs = vector_db.get(where={"file_hash": file_hash})
    if existing_docs and existing_docs.get('ids'):
        return {"message": "⚠️ 该文件内容已存在，无需重复上传！"}

    # ========== 4. 永久保存到 data 目录 ==========
    os.makedirs("./data", exist_ok=True)
    file_location = os.path.join("./data", file.filename)
    with open(file_location, "wb+") as buffer:
        buffer.write(file_content)

    # ========== 5. 读取、切分、入库 ==========
    # ========== 5. 读取、切分、入库 ==========
    from langchain_core.documents import Document
    try:
        if file.filename.endswith((".png", ".jpg", ".jpeg")):
            # ⭐ 图片处理：直接 OCR 提取文字
            print(f"📷 检测到图片，启动 OCR 识别...")
            ocr_text = extract_text_from_image(file_location)
            if not ocr_text.strip():
                return {"error": "❌ 图片中未识别到文字"}
            documents = [Document(page_content=ocr_text, metadata={"source": file.filename})]

        elif file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_location)
            documents = loader.load()
            # ⭐ 扫描版 PDF 处理：如果提取不出字，自动转 OCR
            documents = ocr_pdf_if_empty(file_location, documents)

        elif file.filename.endswith(".txt"):
            loader = TextLoader(file_location, encoding="utf-8")
            documents = loader.load()
        elif file.filename.endswith(".docx"):
            documents = smart_docx_loader(file_location, file.filename)
        # 打标签
        for doc in documents:
            doc.metadata["file_hash"] = file_hash
            doc.metadata["source_file"] = file.filename

        # ========== 父子分块（Small-to-Big） ==========
        # 父块：1500字，用于给大模型看完整上下文
        # 子块：400字，用于精准向量检索
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
        for doc in documents:
            # ⭐ 修复：用 file.filename 作为 source_file
            source_file = file.filename

            # 1. 先把整篇文档切成父块
            parent_chunks = parent_splitter.split_documents([doc])
            for p_idx, parent_chunk in enumerate(parent_chunks):
                # 生成父块唯一 ID
                content_hash = hashlib.md5(parent_chunk.page_content.encode("utf-8")).hexdigest()[:8]
                parent_id = f"{source_file}__p{p_idx}__{content_hash}"

                # 2. 再对父块切子块
                child_chunks = child_splitter.split_documents([parent_chunk])
                for child_chunk in child_chunks:
                    child_chunk.metadata["parent_id"] = parent_id
                    child_chunk.metadata["parent_content"] = parent_chunk.page_content
                    chunks.append(child_chunk)

        # 给每个子块加来源前缀
        for chunk in chunks:
            source_file = chunk.metadata.get("source_file", "未知文件")
            chunk.page_content = f"【文档来源】：{source_file}\n{chunk.page_content}"

        # 入库
        if chunks:
            vector_db.add_documents(chunks)
            # ⭐ 增量更新 BM25 索引（从 Chroma 全量重建，最稳）
            try:
                global bm25_retriever
                all_data_new = vector_db.get()
                if all_data_new and all_data_new.get('documents'):
                    all_docs_new = [
                        Document(page_content=text, metadata=meta)
                        for text, meta in zip(all_data_new['documents'], all_data_new['metadatas'])
                    ]
                    bm25_retriever = BM25Retriever.from_documents(all_docs_new)
                    bm25_retriever.k = 5
                    print(f"✅ BM25 索引已更新：共 {len(all_docs_new)} 个片段")
            except Exception as e:
                print(f"⚠️ BM25 更新失败: {e}")
            # ============================================================
            # ⭐ 智能化：用 LLM 语义判断这份文档是否包含"可执行的审查规则"
            # 不再依赖文件名关键词，而是看内容本身
            # ============================================================
            print(f"🔍 正在判断文档是否包含审查规则...")

            # 取文档前 3000 字 + 标题作为判断依据
            judge_content = "\n".join([c.page_content for c in chunks])[:3000]
            judge_prompt = f"""你是一个高校行政文档分析专家。请判断以下文档**是否包含可执行的审查规则或审核标准**。

            【判断标准】：
            - 如果文档描述了"申请条件"、"审核要求"、"评定标准"、"必须满足的条件"、"合规要求"等可用于**核对材料是否合格**的规则 → 回复 YES
            - 如果文档只是"通知"、"公告"、"温馨提示"、"会议记录"、"新闻稿"等纯信息性内容，没有审核标准 → 回复 NO

            【文档标题】：{file.filename}
            【文档内容】：
            {judge_content}

            只输出 YES 或 NO，不要任何解释。
            """
            try:
                judge_res = await client.chat.completions.create(
                    model="ep-20260919155615-h2mfv",
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,
                    timeout=10.0
                )
                judge_verdict = judge_res.choices[0].message.content.strip().upper()
                is_rule_doc = "YES" in judge_verdict
                print(f"   判断结果: {judge_verdict} → {'包含规则' if is_rule_doc else '普通文档'}")
            except Exception as e:
                print(f"   ⚠️ 语义判断失败，降级为关键词匹配: {e}")
                rule_keywords = ["办法", "规定", "细则", "条例", "章程", "制度", "准则", "标准", "规范", "说明", "要求"]
                is_rule_doc = any(kw in file.filename for kw in rule_keywords)

            rule_msg = ""
            if is_rule_doc:
                print(f"📖 检测到规则类文档，正在抽取规则...")

                # 1. 自动识别/创建领域（允许 LLM 发现新领域）
                domain_prompt = f"""请判断以下文档属于哪个业务领域。

            【已知领域】：{', '.join(DOMAIN_TEMPLATES.keys())}

            【判断规则】：
            1. 如果文档属于上述已知领域之一，直接输出该领域名称。
            2. 如果不属于任何一个已知领域，请根据文档内容**创建一个新的、简洁的领域名称**（4-8个字，如"团员推优"、"社团管理"、"转专业申请"）。
            3. 只输出领域名称，不要任何解释。

            【文档标题】：{file.filename}
            【文档开头】：{chunks[0].page_content[:800]}

            领域："""
                domain_res = await client.chat.completions.create(
                    model="ep-20260919155615-h2mfv",
                    messages=[{"role": "user", "content": domain_prompt}],
                    temperature=0.0
                )
                matched_domain = domain_res.choices[0].message.content.strip()
                print(f"   ✅ 识别到领域：【{matched_domain}】")

                # 2. 抽取规则
                doc_content = "\n".join([c.page_content for c in chunks])
                new_rules = await extract_rules_from_content(doc_content, file.filename, matched_domain)

                # 3. 更新 auto_rules.json（v2.0 字典结构）
                if new_rules:
                    if os.path.exists("auto_rules.json"):
                        with open("auto_rules.json", "r", encoding="utf-8") as f:
                            rules_config = json.load(f)
                    else:
                        rules_config = {"version": "2.0", "domains": {}, "global_rules": []}

                    # 兼容旧结构，强制升级为字典
                    if "domains" not in rules_config or not isinstance(rules_config["domains"], dict):
                        rules_config["domains"] = {}
                    if "global_rules" not in rules_config:
                        rules_config["global_rules"] = []

                    if matched_domain not in rules_config["domains"]:
                        rules_config["domains"][matched_domain] = {"rules": [], "source_docs": []}

                    # ⭐ 去重：只添加 id 不重复的规则
                    existing_ids = {r["id"] for r in rules_config["domains"][matched_domain]["rules"]}
                    deduped_rules = [r for r in new_rules if r.get("id") not in existing_ids]

                    rules_config["domains"][matched_domain]["rules"].extend(deduped_rules)
                    if file.filename not in rules_config["domains"][matched_domain]["source_docs"]:
                        rules_config["domains"][matched_domain]["source_docs"].append(file.filename)

                    with open("auto_rules.json", "w", encoding="utf-8") as f:
                        json.dump(rules_config, f, ensure_ascii=False, indent=2)

                    rule_msg = f"，并自动抽取了 {len(deduped_rules)} 条【{matched_domain}】规则到规则库"
                    print(f"✅ 规则库已更新：新增 {len(deduped_rules)} 条规则（去重后）")

            return {"message": f"✅ 上传成功！已添加 {len(chunks)} 个文本片段到知识库{rule_msg}"}
        else:
            return {"message": "文件内容为空，未添加片段"}

    # ⭐ 把 finally 换成 except，当解析报错时，给出提示，而不是删除文件！
    except Exception as e:
        print(f"❌ 解析文件失败: {e}")
        return {"error": f"文件解析失败，可能格式不正确或编码有问题"}


@app.post("/review")
async def review_file(file: UploadFile = File(...)):
    import json as _json
    import re

    file_location = f"./temp_{file.filename}"
    with open(file_location, "wb+") as buffer:
        buffer.write(await file.read())

    try:
        # ========== 1. 读取待审材料 ==========
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_location)
        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(file_location)
        else:
            loader = TextLoader(file_location, encoding="utf-8")
        documents = loader.load()
        doc_content = "\n".join([d.page_content for d in documents])

        # ========== 2. 自动识别材料类型，匹配规则域 ==========
        # ========== 2. 自动识别材料类型（领域 + 子类型） ==========
        print("⏳ 正在识别材料类型...")
        domain_prompt = f"""请分析以下待审材料，输出结构化识别结果。

        【材料内容】：
        {doc_content[:1000]}

        【已知领域】：{list(DOMAIN_TEMPLATES.keys())}

        【识别任务】：
        1. domain：判断材料属于哪个业务领域（从已知领域选，或新建一个简洁领域名）
        2. scene：判断材料的**具体子类型**（从下面选一个）：
           - "票根"：投票票根、选票
           - "申请表"：申请类材料
           - "推荐表"：推荐类材料
           - "会议记录"：会议记录、会议纪要
           - "资格审查表"：资格审查相关
           - "通知公告"：通知、公告
           - "通用"：无法判断的
        3. key_fields：从材料中提取 3-5 个关键字段（如人名、日期、学号）

        【输出格式】（只输出合法 JSON）：
        {{"domain": "团员推优", "scene": "票根", "key_fields": ["姓名", "赞成票数", "日期"]}}
        """
        domain_res = await client.chat.completions.create(
            model="ep-20260919155615-h2mfv",
            messages=[{"role": "user", "content": domain_prompt}],
            temperature=0.0
        )
        domain_raw = domain_res.choices[0].message.content.strip()
        if domain_raw.startswith("```"):
            domain_raw = domain_raw.strip("`").replace("json\n", "", 1)

        try:
            domain_info = _json.loads(domain_raw)
            matched_domain = domain_info.get("domain", "通用审查")
            matched_scene = domain_info.get("scene", "通用")
        except Exception as e:
            print(f"⚠️ 领域识别解析失败: {e}")
            matched_domain = "通用审查"
            matched_scene = "通用"

        print(f"✅ 识别到领域：【{matched_domain}】，子类型：【{matched_scene}】")

        # ========== 3. 读取 auto_rules.json，严格领域匹配 ==========
        if not os.path.exists("auto_rules.json"):
            return {"error": "规则库未初始化，请先上传规章制度文件"}

        with open("auto_rules.json", "r", encoding="utf-8") as f:
            rules_config = _json.load(f)

        # ⭐ 严格领域匹配（不再无脑兜底）
        matched_rules = route_rules(matched_domain, matched_scene, rules_config)
        print(f"✅ 最终匹配到 {len(matched_rules)} 条规则")

        # ========== 4. ⭐ 动态事实抽取：根据匹配到的规则反推字段 ==========
        print("⏳ 正在抽取事实清单（动态 schema）...")

        # 从规则里收集所有 field
        required_fields = sorted(set(
            rule.get("field", "") for rule in matched_rules if rule.get("field")
        ))

        # 兜底：如果规则里没 field，用默认几个
        if not required_fields:
            required_fields = ["姓名", "学号", "学院", "班级", "职位", "日期", "绩点"]

        # 排除一些不适合让 LLM 直接抽取的字段（比如"正文"、"全文"这种关键词型）
        required_fields = [f for f in required_fields if f not in ["正文", "全文", "_full_text"]]

        # 构造动态 JSON schema
        schema_lines = [f'    "{f}": ""' for f in required_fields]
        schema_str = "{\n" + ",\n".join(schema_lines) + "\n}"

        print(f"   📋 动态字段: {required_fields}")

        fact_prompt = f"""请从以下材料中抽取结构化事实。只输出合法 JSON，不要任何解释。
        如果某个字段在材料中找不到，请填写空字符串 ""。

        【材料】：
        {doc_content}

        【输出格式】：
        {schema_str}
        """
        fact_res = await client.chat.completions.create(
            model="ep-20260919155615-h2mfv",
            messages=[{"role": "user", "content": fact_prompt}],
            temperature=0.0
        )
        fact_raw = fact_res.choices[0].message.content.strip()
        if fact_raw.startswith("```"):
            fact_raw = fact_raw.strip("`").replace("json\n", "", 1)

        try:
            facts = _json.loads(fact_raw)
        except:
            facts = {}
        facts["_full_text"] = doc_content

        # ========== 5. 硬规则校验 ==========
        print("⏳ 正在执行硬规则校验...")
        hard_results = execute_hard_rules(facts, matched_rules)
        passed_count = sum(1 for r in hard_results if r["passed"])
        print(f"✅ 硬规则：{passed_count}/{len(hard_results)} 通过")

        # ========== 6. 软审查 ==========
        print("⏳ 正在执行软审查...")
        soft_prompt = f"""你是一个高校行政材料审查员。请对以下材料进行软性审查，只关注表述规范性和逻辑一致性。

【材料】：
{doc_content}

【重要提醒】：
- 请严格区分"严重问题"和"建议性优化"。
- 如果只是"排版可以更好""措辞可以更规范"这种小瑕疵，不要列为"问题"，可以放在"建议"里。
- 满分 100 分对应"完全可以直接使用"，80 分对应"小修即可"，60 分对应"需要明显修改"。
- 请以"这份材料能不能直接提交"为标准评分，而不是"它能不能变得更好"。

【输出格式】（只输出合法 JSON）：
{{
    "表述规范性": {{"评分": <0-100>, "问题": []}},
    "逻辑一致性": {{"评分": <0-100>, "问题": []}},
    "遗漏项": []
}}
"""
        soft_res = await client.chat.completions.create(
            model="ep-20260919155615-h2mfv",
            messages=[{"role": "user", "content": soft_prompt}],
            temperature=0.1
        )
        soft_raw = soft_res.choices[0].message.content.strip()
        if soft_raw.startswith("```"):
            soft_raw = soft_raw.strip("`").replace("json\n", "", 1)
        try:
            soft_result = _json.loads(soft_raw)
        except:
            soft_result = {}

        # ========== 7. 综合评分 ==========
        # ⭐ 三档扣分制
        # ⭐ 三档扣分制
        if hard_results:
            base_score = 100
            deduction = 0
            critical_count = 0
            error_count = 0
            warning_count = 0

            for r in hard_results:
                if not r["passed"]:
                    sev = r["severity"]
                    if sev == "critical":
                        deduction += 30
                        critical_count += 1
                    elif sev == "error":
                        deduction += 10
                        error_count += 1
                    else:
                        deduction += 3
                        warning_count += 1

            hard_score = max(0, base_score - deduction)
            print(
                f"   💯 硬规则扣分：致命 {critical_count} × 30 + 严重 {error_count} × 10 + 建议 {warning_count} × 3 = {deduction} 分")
            print(f"   最终硬规则得分：{hard_score}/100")
        else:
            hard_score = 100
        soft_score = int(sum([
            soft_result.get("表述规范性", {}).get("评分", 100),
            soft_result.get("逻辑一致性", {}).get("评分", 100)
        ]) / 2)
        total_score = int(hard_score * 0.6 + soft_score * 0.4)

        if total_score >= 90:
            grade = "优秀"
        elif total_score >= 75:
            grade = "合格"
        elif total_score >= 60:
            grade = "需修改"
        else:
            grade = "不合格"

        return {
            "review_result": {
                "材料类型": matched_domain,
                "总评分": total_score,
                "等级": grade,
                "硬规则得分": hard_score,
                "软审查得分": soft_score,
                "事实清单": {k: v for k, v in facts.items() if k != "_full_text"},
                "硬规则校验": hard_results,
                "软审查": soft_result,
                "适用规则数": len(matched_rules)
            }
        }

    finally:
        if os.path.exists(file_location):
            os.remove(file_location)


@app.get("/preview/{filename:path}")
async def preview_file(filename: str):
    # 安全检查：防止路径穿越攻击
    safe_filename = os.path.basename(filename)
    file_path = os.path.join("./data", safe_filename)  # 假设文件都在 data 目录里

    if not os.path.exists(file_path):
        return {"error": "文件不存在"}

    # 返回文件给浏览器预览
    return FileResponse(file_path)
app.mount("/", StaticFiles(directory="static", html=True), name="static")#这行代码是整个后端项目的“门面装修”，第一先给地址
#第二个装修店面
#uvicorn main:app --reload