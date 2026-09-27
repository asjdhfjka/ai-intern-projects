"""
一键重建规则库
用途：扫描 data 文件夹里的规章制度文档，重新抽取规则到 auto_rules.json
使用：python rebuild_rules.py
场景：规则库需要升级 schema 时，不用重新上传文件
"""
import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

# ⭐ 规则抽取的领域模板
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


async def judge_is_rule_doc(doc_content: str, filename: str) -> bool:
    """用 LLM 语义判断文档是否包含可执行的审查规则"""
    prompt = f"""判断以下文档是否包含可执行的审查规则或审核标准。

【判断标准】：
- 包含"申请条件"、"审核要求"、"评定标准"、"合规要求" → YES
- 只是"通知"、"公告"、"温馨提示"、"新闻稿" → NO

【文档标题】：{filename}
【文档内容】：
{doc_content[:3000]}

只输出 YES 或 NO。
"""
    try:
        res = await client.chat.completions.create(
            model="ep-20260919155615-h2mfv",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return "YES" in res.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"   判断失败: {e}")
        return False


async def identify_domain(doc_content: str, filename: str) -> str:
    """识别文档所属领域"""
    prompt = f"""请判断以下文档属于哪个业务领域。

【已知领域】：{', '.join(DOMAIN_TEMPLATES.keys())}

【规则】：
1. 属于已知领域就直接输出该名称
2. 不属于任何已知领域，就创建一个新的简洁领域名（4-8字）

只输出领域名称，不要解释。

【文档标题】：{filename}
【文档开头】：{doc_content[:800]}

领域："""
    try:
        res = await client.chat.completions.create(
            model="ep-20260919155615-h2mfv",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"   领域识别失败: {e}")
        return "通用审查"


async def extract_rules(doc_content: str, doc_title: str, domain: str) -> list:
    """抽取结构化规则（带 scene 字段）"""
    prompt = f"""你是高校行政规章制度分析专家。请从以下文档中抽取可执行的结构化审查规则。

【文档标题】：{doc_title}
【文档领域】：{domain}
【文档正文】：
{doc_content[:6000]}

【规则 Schema（强制）】：
每条规则必须包含以下 8 个字段：
1. id：唯一标识，格式 "{domain}_R001"
2. name：规则名（10-20字）
3. type：required / regex / range / keyword / enum 之一
4. field：被校验的字段名
5. severity：严重程度，必须是以下三档之一：
   - "critical"：致命问题（缺签名、缺盖章、票数不够、资格审查不过），扣 30 分
   - "error"：严重问题（格式错误、信息不全、逻辑不一致），扣 10 分
   - "warning"：建议问题（措辞不规范、排版不美观），扣 3 分
   
   【判断标准】：
   - 如果这条规则不通过会导致材料**完全无法使用** → critical
   - 如果会导致**需要退回修改** → error
   - 如果只是**可以更好但不影响使用** → warning
   
   【判断标准】：
   - 如果这条规则不通过会导致材料**完全无法使用** → critical
   - 如果会导致**需要退回修改** → error
   - 如果只是**可以更好但不影响使用** → warning
6. message：未通过提示
7. scene：⭐ 适用材料类型（数组）
   可选值：["票根", "选票", "申请表", "推荐表", "会议记录", "资格审查", "通知公告", "通用"]
8. condition：（可选）额外适用条件

【scene 判断指引】：
- 提到"票根"、"选票"、"唱票" → ["票根"] 或 ["选票"]
- 提到"申请表"、"申请理由" → ["申请表"]
- 提到"入团满一年"、"无挂科"等资格条件 → ["资格审查"]
- 提到"黑板"、"发言时长"、"会议流程" → ["会议记录"]
- 适用于所有材料的（如日期格式） → ["通用"]

【关于日期格式规则的理解】：
- 原文说"日期写09日而不是9日"，请理解为"日期必须是两位数字"。
- 08日、09日、15日、31日 **全部合法**。
- 只有"8日""9日"这种一位数字才是错的。
- 正则应该用：^(0[1-9]|[12][0-9]|3[01])日$，覆盖 01-31 日。
- message 里请多举几个例子，避免用户误解成"只能写09"。

【可审查性过滤】：
以下类型规则**不要抽取**，因为无法从材料本身校验：
- 流程类规则：如"票根要先发监督干事检查"（票根本身看不出来）
- 操作类规则：如"增行/删行"（这是操作动作，不是材料内容）
- 会议类规则：如"团支书要提前到场"（无法从材料校验）

【非常重要的过滤要求】：
1. **只抽取能从材料本身校验的规则**。
   - ✅ 可校验："票根必须有监票人签名"（能看签名栏）
   - ❌ 不可校验："票根先发监督干事检查"（这是流程要求，票根本身看不出来）
   
2. **只抽取硬性要求，不抽取建议性规范**。
   - ✅ 硬性："入团满一年"
   - ❌ 建议："建议日期格式写成两位数字"（如果原文用"建议"、"最好"等措辞）
   
3. **排除会议流程类规则**。会议流程、拍照数量、人员到场顺序等，不属于"材料审查"的范畴。

【输出格式】（只输出合法 JSON 数组）：
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
    }}
]

【要求】：
- scene 必须是非空数组，不能省略！
- 至少抽 5 条，最多 50 条
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
        print(f"   ❌ 规则抽取失败: {e}")
        return []


async def main():
    data_dir = "./data"
    if not os.path.exists(data_dir):
        print("❌ data 文件夹不存在")
        return

    print(f"📂 扫描 {data_dir} 文件夹...\n")

    # 1. 遍历所有文件
    all_files = [f for f in os.listdir(data_dir) if f.endswith((".txt", ".pdf", ".docx"))]
    print(f"找到 {len(all_files)} 个文档，开始判断哪些是规则文档...\n")

    # 2. 逐个判断 + 抽取
    rules_config = {"version": "2.0", "generated_at": "", "domains": {}, "global_rules": []}

    for i, filename in enumerate(all_files, 1):
        file_path = os.path.join(data_dir, filename)

        # 读取内容
        try:
            if filename.endswith(".pdf"):
                docs = PyPDFLoader(file_path).load()
            elif filename.endswith(".docx"):
                docs = Docx2txtLoader(file_path).load()
            else:
                docs = TextLoader(file_path, encoding="utf-8").load()
            content = "\n".join([d.page_content for d in docs])
        except Exception as e:
            print(f"[{i}/{len(all_files)}] ❌ 读取失败 {filename}: {e}")
            continue

        # 判断是否是规则文档
        print(f"[{i}/{len(all_files)}] 🔍 {filename}")
        is_rule = await judge_is_rule_doc(content, filename)
        if not is_rule:
            print(f"   ⏭️ 跳过（非规则文档）")
            continue

        # 识别领域
        domain = await identify_domain(content, filename)
        print(f"   ✅ 识别领域：【{domain}】")

        # 抽取规则
        rules = await extract_rules(content, filename, domain)
        if not rules:
            print(f"   ⚠️ 未抽到规则")
            continue

        print(f"   ✅ 抽取到 {len(rules)} 条规则")

        # 合并到配置
        if domain not in rules_config["domains"]:
            rules_config["domains"][domain] = {"rules": [], "source_docs": []}
        rules_config["domains"][domain]["rules"].extend(rules)
        rules_config["domains"][domain]["source_docs"].append(filename)

    # 3. 保存
    from datetime import datetime
    rules_config["generated_at"] = str(datetime.now())

    with open("auto_rules.json", "w", encoding="utf-8") as f:
        json.dump(rules_config, f, ensure_ascii=False, indent=2)

    total = sum(len(d["rules"]) for d in rules_config["domains"].values())
    print(f"\n🎉 重建完成！")
    print(f"📊 共 {len(rules_config['domains'])} 个领域，{total} 条规则")
    print(f"📄 已保存到 auto_rules.json")


if __name__ == "__main__":
    asyncio.run(main())