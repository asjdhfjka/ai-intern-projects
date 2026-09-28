"""
工大智政系统批量测试 + 树状图生成
使用：
    1. 先启动 main.py 服务（uvicorn main:app --reload）
    2. 运行本脚本：python test_all.py
    3. 生成 test_tree.png 和 test_results.json
"""
import requests
import json
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============ 中文字体 ============
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 测试问题清单 ============
# 格式：(问题, 期望关键词列表, 测试类型)
TEST_CASES = {
    "计算机等级考试": [
        ("2026年上半年全国计算机等级考试什么时候报名？", ["报名时间", "3月", "上半年"], "时间提取"),
        ("2026年下半年计算机等级考试报名条件是什么？", ["报名", "上传照片", "信息"], "条件提取"),
        ("计算机等级考试报名费是多少？", ["费用", "元", "报名费"], "数字提取"),
        ("计算机二级考试有哪些科目可以报？", ["科目", "二级"], "列表提取"),
        ("上半年和下半年计算机等级考试的报名时间分别是什么？", ["上半年", "下半年"], "多文件对比"),
        ("计算机等级考试准考证怎么打印？", ["打印", "准考证"], "流程型"),
        ("计算机等级考试成绩多久可以查？", ["成绩", "查询", "发布"], "时间提取"),
        ("我们学校计算机等级考试在哪里报名？", ["报名", "网址", "入口"], "细节提取"),
    ],
    "团员推优/评议": [
        ("团员推优会议的流程是什么？", ["发言", "投票", "唱票"], "流程型"),
        ("推优需要多少票才能通过？", ["赞成", "半数", "票"], "数字提取"),
        ("入团满多久可以参加推优？", ["入团", "满一年", "一年"], "条件提取"),
        ("团员评议会议记录要写哪些内容？", ["评议", "会议", "记录"], "细节提取"),
        ("推优会议上监票人是怎么产生的？", ["监票", "唱票"], "细节提取"),
        ("何华儒是哪个班的？担任什么职位？", ["计算机", "9班", "学习委员"], "人名检索"),
    ],
    "奖学金申请": [
        ("优秀学生奖学金申请表需要哪些材料？", ["申请表", "材料"], "列表提取"),
        ("申请优秀学生奖学金需要满足什么条件？", ["条件", "绩点"], "条件提取"),
        ("奖学金申请表里'绩点'一栏怎么填？", ["绩点", "填写"], "表格理解"),
        ("奖学金评选的流程是什么？", ["流程", "评审", "公示"], "流程型"),
        ("有挂科还能申请奖学金吗？", ["挂科", "不能", "不符合"], "边界条件"),
    ],
    "其他行政通知": [
        ("'十五五'规划建言献策活动怎么参加？", ["建言", "献策", "提交"], "流程型"),
        ("建言献策的截止时间是什么时候？", ["截止", "时间"], "时间提取"),
        ("学校巡视组的巡视时间是多久？", ["巡视", "时间"], "时间提取"),
        ("巡视期间怎么联系巡视组？", ["联系", "电话", "邮箱"], "细节提取"),
        ("巡视公告主要巡视哪些内容？", ["巡视", "内容"], "内容提取"),
        ("2026届那个学生邀请函是什么活动？", ["邀请函", "招聘", "活动"], "事实型"),
        ("参加那个学生活动需要什么条件？", ["条件", "要求"], "条件提取"),
        ("那个活动怎么报名参加？", ["报名", "参加"], "流程型"),
    ],
    "综合/边界测试": [
        ("四级考试报名费怎么交？", ["未提及", "未找到", "无法回答"], "条件提取"),
        ("啥时候报名啊", ["计算机", "报名"], "代词消解"),
        ("奖学金和推优有什么关系？", ["奖学金", "推优"], "跨文件理解"),
    ],
}
# TEST_CASES = {
#     "计算机等级考试": [
#         ("计算机一级考试考什么内容？", ["一级", "内容"], "考试内容细节"),
#         ("计算机等级考试可以跨级报考吗？", ["跨级", "级别"], "报考规则边界"),
#         ("计算机等级考试证书有效期是多久？", ["证书", "有效期"], "证书相关"),
#         ("上半年计算机等级考试的准考证什么时候可以打印？", ["3月23日", "打印", "准考证"], "时间提取"),
#         ("计算机等级考试缺考了下次还能报吗？", ["缺考", "再报"], "缺考规则"),
#         ("计算机等级考试成绩不合格可以补考吗？", ["补考", "不合格"], "补考政策"),
#     ],
#     "团员推优/评议": [
#         ("推优会议需要多少人参加才有效？", ["实到", "2/3", "0.8"], "到会人数要求"),
#         ("团员评议的等次有哪几种？", ["优秀", "合格"], "评议等级"),
#         ("推优结果需要公示吗？公示多久？", ["公示"], "公示要求"),
#         ("群众代表可以参加推优会议吗？", ["团员", "参会"], "参会人员范围"),
#         ("推优材料需要上交到哪里？", ["监督干事", "保存"], "材料提交去向"),
#         ("团员评议多久开展一次？", ["学期", "学年", "一次"], "频次要求"),
#         ("推优会议上不同意票怎么算？", ["反对", "弃权", "票"], "票数计算规则"),
#     ],
#     "奖学金申请": [
#         ("优秀学生奖学金分几个等级？", ["等级", "一等", "二等", "三等"], "等级分类"),
#         ("奖学金评选需要班级民主投票吗？", ["投票", "民主", "评选"], "评选方式"),
#         ("同一学年可以同时申请多个奖学金吗？", ["同时", "多个", "互斥"], "互斥规则"),
#         ("奖学金什么时候发放？", ["发放", "时间"], "发放时间"),
#         ("奖学金申请表里'主要事迹'怎么写？", ["主要事迹", "填写"], "填写指导"),
#         ("获得奖学金后如果挂科了会取消吗？", ["取消", "挂科"], "后续资格"),
#     ],
#     "其他行政通知": [
#         ("建言献策活动有没有奖励？", ["奖励", "奖"], "奖励机制"),
#         ("建言献策可以以团队名义提交吗？", ["团队", "个人", "名义"], "提交形式"),
#         ("巡视组受理哪些方面的问题？", ["受理", "举报", "反映"], "受理范围"),
#         ("巡视期间可以通过什么方式反映问题？", ["电话", "邮寄", "扫码"], "反映渠道"),
#         ("2026届那个学生活动的时间和地点是什么？", ["3月28日", "图书馆"], "活动信息"),
#         ("参加那个活动需要准备什么材料？", ["材料", "简历", "准备"], "准备事项"),
#     ],
#     "辅导员/管理视角": [
#         ("班里要组织推优，我作为辅导员第一步要做什么？", ["统计", "名单", "智慧团建"], "流程起点"),
#         ("收集上来的奖学金申请表，要检查哪些东西？", ["检查", "签名", "盖章"], "审查视角"),
#         ("巡视公告出来后，学院需要配合做什么？", ["配合", "巡视", "提供"], "执行视角"),
#     ],
#     "综合/边界测试": [
#         ("图书馆怎么预约座位？", ["未提及", "无法回答", "未找到"], "拒答能力"),
#         ("那个表怎么填", ["奖学金", "申请表", "填写"], "代词消解"),
#     ],
# }

def ask_question(question: str, timeout: int = 60) -> str:
    """调用 /chat/stream 接口"""
    url = "http://127.0.0.1:8000/chat/stream"
    params = {"user_input": question, "history": ""}
    try:
        resp = requests.post(url, params=params, stream=True, timeout=timeout)
        answer = ""
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                answer += chunk.decode('utf-8', errors='ignore')
        # 过滤 __SOURCES__...__END__
        if "__SOURCES__" in answer:
            start = answer.find("__SOURCES__")
            end = answer.find("__END__", start)
            if end > 0:
                answer = answer[:start] + answer[end + 7:]
        return answer.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def evaluate(answer: str, keywords: list, q_type: str) -> str:
    """简单评估：返回 passed / partial / rejected / failed"""
    if not answer or answer.startswith("[ERROR]"):
        return "failed"

    # ⭐ 拒答检测（更严格：只在回答很短且明确拒答时才判拒答）
    rejected_phrases = ["无法回答", "无法准确回答", "未找到", "资料中没有"]
    # 如果回答长度 > 200 字，说明系统给了内容，不算拒答
    is_short = len(answer) < 200
    is_rejected = is_short and any(p in answer for p in rejected_phrases)

    if q_type == "拒答能力":
        return "passed" if is_rejected else "failed"

    # ⭐ 关键词命中优先于拒答判断
    hit = sum(1 for kw in keywords if kw in answer)
    if hit >= max(1, len(keywords) // 2):
        return "passed"
    elif hit >= 1:
        return "partial"
    elif is_rejected:
        return "rejected"
    else:
        return "failed"


def run_all_tests() -> list:
    """执行所有测试"""
    results = []
    total = sum(len(qs) for qs in TEST_CASES.values())
    counter = 0
    for category, questions in TEST_CASES.items():
        print(f"\n{'=' * 60}")
        print(f"📂 分类：{category}")
        print(f"{'=' * 60}")
        for question, keywords, q_type in questions:
            counter += 1
            print(f"\n[{counter}/{total}] ❓ {question}")
            answer = ask_question(question)
            status = evaluate(answer, keywords, q_type)
            status_emoji = {
                "passed": "✅", "partial": "🟡",
                "rejected": "⚪", "failed": "❌"
            }[status]
            print(f"   {status_emoji} 状态：{status}")
            print(f"   💬 回答（前150字）：{answer[:150]}")
            results.append({
                "category": category,
                "question": question,
                "answer": answer,
                "status": status,
                "type": q_type,
                "keywords": keywords,
            })
            time.sleep(1)  # 避免 API 过载
    return results


def draw_tree(results: list, output: str = "test_tree.png"):
    """生成横向树状图"""
    # 计算叶子节点的 y 坐标
    leaf_y = {}
    current_y = 0
    for cat in TEST_CASES:
        for q, _, _ in TEST_CASES[cat]:
            leaf_y[q] = current_y
            current_y += 1
    total_leaves = current_y

    # 分类节点的 y 坐标（取叶子平均）
    cat_y = {}
    for cat in TEST_CASES:
        ys = [leaf_y[q] for q, _, _ in TEST_CASES[cat]]
        cat_y[cat] = sum(ys) / len(ys)

    root_y = total_leaves / 2

    # 状态颜色
    status_colors = {
        "passed": "#28a745",    # 绿
        "partial": "#ffc107",   # 黄
        "rejected": "#6c757d",  # 灰
        "failed": "#dc3545",    # 红
    }

    # 建索引
    result_by_q = {r["question"]: r for r in results}

    # 画布
    fig, ax = plt.subplots(figsize=(22, 18))

    # ---- 根节点 ----
    ax.plot(0, root_y, 'o', markersize=40, color='#212529', zorder=3)
    ax.text(0, root_y + 1.2, "工大智政\n测试总览", ha='center', va='bottom',
            fontsize=14, fontweight='bold')

    # ---- 分类节点 + 连线 ----
    for cat in TEST_CASES:
        cy = cat_y[cat]
        ax.plot(1, cy, 'o', markersize=25, color='#495057', zorder=3)
        ax.text(1, cy + 0.8, cat, ha='center', va='bottom',
                fontsize=12, fontweight='bold')
        ax.plot([0.15, 0.85], [root_y, cy], '-', color='#adb5bd',
                linewidth=2, zorder=1)

    # ---- 叶子节点 + 连线 ----
    for cat in TEST_CASES:
        cy = cat_y[cat]
        for q, _, q_type in TEST_CASES[cat]:
            qy = leaf_y[q]
            status = result_by_q[q]["status"]
            color = status_colors[status]
            ax.plot(2, qy, 'o', markersize=14, color=color, zorder=3)
            # 问题文本（截断到 28 字）
            short_q = q if len(q) <= 28 else q[:26] + "..."
            ax.text(2.1, qy, short_q, fontsize=9.5, va='center',
                    color='#212529')
            # 测试类型标签
            ax.text(4.5, qy, f"[{q_type}]", fontsize=8, va='center',
                    color='#6c757d', style='italic')
            # 从分类到叶子的连线
            ax.annotate("", xy=(1.9, qy), xytext=(1.1, cy),
                        arrowprops=dict(arrowstyle='-', color='#ced4da',
                                        linewidth=0.8))

    # ---- 图例 ----
    legend_elements = [
        mpatches.Patch(color='#28a745', label='通过'),
        mpatches.Patch(color='#ffc107', label='部分通过'),
        mpatches.Patch(color='#6c757d', label='合理拒答'),
        mpatches.Patch(color='#dc3545', label='失败'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=11, frameon=True)

    # ---- 统计信息 ----
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "passed")
    partial = sum(1 for r in results if r["status"] == "partial")
    rejected = sum(1 for r in results if r["status"] == "rejected")
    failed = sum(1 for r in results if r["status"] == "failed")
    stats_text = f"""统计
    总计：{total} 题
    通过：{passed} ({passed / total * 100:.0f}%)
    部分：{partial} ({partial / total * 100:.0f}%)
    拒答：{rejected} ({rejected / total * 100:.0f}%)
    失败：{failed} ({failed / total * 100:.0f}%)"""
    ax.text(5, total_leaves * 0.85, stats_text, fontsize=12,
            va='top', bbox=dict(boxstyle='round,pad=0.8',
                                facecolor='#f8f9fa', edgecolor='#dee2e6'))

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.5, total_leaves + 1.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n🌳 树状图已保存：{output}")
    plt.close()


def main():
    print("🚀 工大智政系统批量测试")
    print("=" * 60)
    print("⚠️  请确认 main.py 服务已启动 (uvicorn main:app --reload)")
    print("=" * 60)

    # 检查服务是否在线
    try:
        r = requests.get("http://127.0.0.1:8000/", timeout=5)
        print(f"✅ 服务在线 (HTTP {r.status_code})\n")
    except Exception as e:
        print(f"❌ 无法连接服务：{e}")
        print("请先启动：uvicorn main:app --reload")
        return

    # 执行测试
    results = run_all_tests()

    # 保存结果
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 测试结果已保存：test_results.json")

    # 生成树状图
    draw_tree(results)

    # 终端统计
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "passed")
    print(f"\n{'=' * 60}")
    print(f"📊 最终成绩：{passed}/{total} = {passed / total * 100:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()