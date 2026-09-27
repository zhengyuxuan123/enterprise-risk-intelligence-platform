from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
W, H = 2000, 1250

BG = "#F6F8FB"
INK = "#172B3A"
MUTED = "#587080"
BORDER = "#B8C5CE"
BLUE = "#2F6FED"
CYAN = "#158E9B"
GREEN = "#2E8B57"
ORANGE = "#D97917"
RED = "#C44536"
PURPLE = "#6F5CC2"
WHITE = "#FFFFFF"

FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")


def font(size, bold=False):
    p = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
    return ImageFont.truetype(str(p), size)


F_TITLE = font(48, True)
F_SUB = font(23)
F_GROUP = font(28, True)
F_NODE = font(25, True)
F_BODY = font(21)
F_SMALL = font(18)
F_EDGE = font(17)


def canvas(title, subtitle):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((70, 46), title, font=F_TITLE, fill=INK)
    d.text((72, 112), subtitle, font=F_SUB, fill=MUTED)
    d.line((70, 158, W - 70, 158), fill=BORDER, width=2)
    return im, d


def wrap(draw, text, fnt, max_width):
    lines = []
    for raw in str(text).split("\n"):
        if not raw:
            lines.append("")
            continue
        line = ""
        for ch in raw:
            test = line + ch
            if draw.textlength(test, font=fnt) <= max_width or not line:
                line = test
            else:
                lines.append(line)
                line = ch
        if line:
            lines.append(line)
    return lines


def box(d, xy, title, body="", color=BLUE, fill=WHITE, radius=8,
        title_color=None, body_color=MUTED, align="center"):
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=color, width=3)
    tc = title_color or color
    title_font = F_NODE
    body_font = F_BODY
    for title_size, body_size in ((25, 21), (23, 19), (21, 17), (19, 16)):
        title_font = font(title_size, True)
        body_font = font(body_size)
        title_lines = wrap(d, title, title_font, x2 - x1 - 28)
        body_lines = wrap(d, body, body_font, x2 - x1 - 32) if body else []
        title_step = title_size + 10
        body_step = body_size + 8
        total = len(title_lines) * title_step + (9 if body_lines else 0) + len(body_lines) * body_step
        if total <= y2 - y1 - 24:
            break
    y = y1 + max(16, ((y2 - y1) - total) / 2)
    for line in title_lines:
        w = d.textlength(line, font=title_font)
        x = x1 + 18 if align == "left" else x1 + (x2 - x1 - w) / 2
        d.text((x, y), line, font=title_font, fill=tc)
        y += title_step
    if body_lines:
        y += 7
        for line in body_lines:
            w = d.textlength(line, font=body_font)
            x = x1 + 18 if align == "left" else x1 + (x2 - x1 - w) / 2
            d.text((x, y), line, font=body_font, fill=body_color)
            y += body_step


def group(d, xy, title, color):
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=8, fill=WHITE, outline=BORDER, width=2)
    d.rectangle((x1, y1, x2, y1 + 48), fill=color)
    d.text((x1 + 18, y1 + 8), title, font=F_GROUP, fill=WHITE)


def arrow(d, start, end, color=MUTED, width=4, label=None, dashed=False):
    x1, y1 = start
    x2, y2 = end
    if dashed:
        steps = 18
        for i in range(0, steps, 2):
            a = i / steps
            b = min((i + 1) / steps, 1)
            d.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a,
                    x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=color, width=width)
    else:
        d.line((x1, y1, x2, y2), fill=color, width=width)
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 15
    pts = [(x2, y2),
           (x2 - size * math.cos(ang - 0.55), y2 - size * math.sin(ang - 0.55)),
           (x2 - size * math.cos(ang + 0.55), y2 - size * math.sin(ang + 0.55))]
    d.polygon(pts, fill=color)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        tw = d.textlength(label, font=F_EDGE)
        d.rounded_rectangle((mx - tw / 2 - 8, my - 16, mx + tw / 2 + 8, my + 12),
                            radius=4, fill=BG)
        d.text((mx - tw / 2, my - 14), label, font=F_EDGE, fill=color)


def route_arrow(d, points, color=MUTED, width=4, label=None, dashed=False):
    for i in range(len(points) - 2):
        p1, p2 = points[i], points[i + 1]
        if dashed:
            arrow(d, p1, p2, color, width, dashed=True)
        else:
            d.line((*p1, *p2), fill=color, width=width)
    arrow(d, points[-2], points[-1], color, width, label=label, dashed=dashed)


def diamond(d, center, size, text, color=ORANGE):
    x, y = center
    pts = [(x, y - size), (x + size * 1.55, y), (x, y + size), (x - size * 1.55, y)]
    d.polygon(pts, fill=WHITE, outline=color)
    d.line(pts + [pts[0]], fill=color, width=3, joint="curve")
    lines = wrap(d, text, F_BODY, size * 2.2)
    yy = y - len(lines) * 15
    for line in lines:
        tw = d.textlength(line, font=F_BODY)
        d.text((x - tw / 2, yy), line, font=F_BODY, fill=color)
        yy += 30


def save(im, name):
    im.save(OUT / name, "PNG", optimize=True)


def architecture():
    im, d = canvas("图1  系统总体架构", "企业经营风险智能分析与决策平台 · Spring Boot + Python Agent + RAG")
    group(d, (70, 190, 1930, 340), "交互层", BLUE)
    box(d, (110, 255, 475, 320), "业务用户 / 管理员", color=BLUE)
    box(d, (635, 235, 1165, 325), "Vue 3 前端", "Pinia · Element Plus · 流式结果渲染", color=BLUE)
    box(d, (1330, 235, 1880, 325), "REST API + SSE", "JWT 请求 · token/meta/done 事件", color=CYAN)
    arrow(d, (475, 287), (635, 287), BLUE)
    arrow(d, (1165, 287), (1330, 287), CYAN)

    group(d, (70, 375, 970, 770), "Spring Boot 业务服务层（8080）", GREEN)
    box(d, (105, 445, 400, 555), "安全与权限", "Spring Security · JWT · RBAC · 数据范围", GREEN)
    box(d, (425, 445, 720, 555), "业务 API", "企业 · 指标 · 投诉 · 竞品 · 风险事件", GREEN)
    box(d, (745, 445, 935, 555), "导入中心", "Excel 校验与落库", GREEN)
    box(d, (105, 590, 400, 710), "风险规则引擎", "阈值/变化率计算 · 风险事件去重", ORANGE)
    box(d, (425, 590, 720, 710), "审计与处置", "分派 · 处理 · 复核 · 关闭 · 留痕", PURPLE)
    box(d, (745, 590, 935, 710), "AI 转发过滤器", "鉴权后转发 /api/ai/* · SSE 透传", CYAN)

    group(d, (1010, 375, 1930, 770), "Python Agent 智能分析层（8081）", PURPLE)
    box(d, (1045, 445, 1315, 555), "FastAPI 接口", "同步/流式分析 · 健康诊断 · RAG 管理", PURPLE)
    box(d, (1340, 445, 1625, 555), "AgentService", "路由 · 预算 · 预取证 · 降级控制", PURPLE)
    box(d, (1650, 445, 1895, 555), "LangGraph", "工具循环 · 最终轮逐 token 输出", PURPLE)
    box(d, (1045, 590, 1315, 710), "工具与证据池", "经营数据 · 知识库 · MCP 联网", CYAN)
    box(d, (1340, 590, 1625, 710), "质量治理", "引用核对/修补 · 复核员 · 输出护栏", RED)
    box(d, (1650, 590, 1895, 710), "记忆与追踪", "短期/长期记忆 · Trace · 评估报告", ORANGE)
    arrow(d, (970, 650), (1045, 500), CYAN, label="HTTP/SSE")
    arrow(d, (1315, 500), (1340, 500), PURPLE)
    arrow(d, (1625, 500), (1650, 500), PURPLE)
    arrow(d, (1772, 555), (1482, 590), PURPLE)

    group(d, (70, 815, 1930, 1170), "数据与外部能力层", ORANGE)
    box(d, (105, 895, 430, 1085), "MySQL 业务数据库", "企业/指标/规则/事件/知识库元数据\nAI 分析、审计、记忆与追踪", GREEN)
    box(d, (485, 895, 820, 1085), "本地 RAG 索引", "SQLite FTS5 + NumPy 向量\nBM25 + KNN + Rerank + MMR", CYAN)
    box(d, (875, 895, 1190, 1085), "业务资料库", "上传文档 · 结构化业务数据\n增量入库 / 重建索引", ORANGE)
    box(d, (1245, 895, 1545, 1085), "外部大模型", "火山方舟 OpenAI Compatible\nDoubao 模型服务", RED)
    box(d, (1600, 895, 1895, 1085), "MCP 网页检索", "stdio 协议边界\nBing/360 并行 · 10 秒硬上限", PURPLE)
    arrow(d, (300, 770), (270, 895), GREEN)
    arrow(d, (1175, 770), (650, 895), CYAN)
    arrow(d, (1060, 895), (820, 990), ORANGE, label="解析/切分")
    arrow(d, (1545, 990), (1600, 990), PURPLE, label="MCP")
    arrow(d, (1395, 895), (1480, 710), RED, label="推理")
    save(im, "fig1-system-architecture.png")


def risk_flow():
    im, d = canvas("图2  风险规则计算流程", "指标写入后触发规则计算；支持同步降级、事件去重与完整处置闭环")
    y = 220
    box(d, (80, y, 330, y + 110), "指标进入系统", "Excel 导入 / API 新增", BLUE)
    box(d, (390, y, 640, y + 110), "指标落库", "metric_record", GREEN)
    diamond(d, (805, y + 55), 56, "MQ 是否启用？", PURPLE)
    box(d, (965, 185, 1240, 285), "RabbitMQ 派发", "MetricRiskListener", PURPLE)
    box(d, (965, 315, 1240, 415), "同步派发", "DirectRiskTaskDispatcher", GREEN)
    box(d, (1340, y, 1910, y + 110), "RiskEngineService.evaluateMetric", "读取指标并进入确定性规则计算", ORANGE)
    arrow(d, (330, 275), (390, 275), BLUE)
    arrow(d, (640, 275), (720, 275), GREEN)
    arrow(d, (890, 250), (965, 235), PURPLE, label="是")
    arrow(d, (890, 305), (965, 365), GREEN, label="否/降级")
    arrow(d, (1240, 235), (1340, 265), PURPLE)
    arrow(d, (1240, 365), (1340, 295), GREEN)

    box(d, (100, 515, 380, 645), "查询启用规则", "metric_code 匹配\nenabled = 1", ORANGE)
    diamond(d, (550, 580), 62, "操作符类型", ORANGE)
    box(d, (720, 475, 1030, 575), "直接阈值比较", "GT / GE / LT / LE", ORANGE)
    box(d, (720, 620, 1030, 730), "变化率计算", "读取上一期指标\nCHANGE_GT / CHANGE_LT", ORANGE)
    diamond(d, (1225, 580), 62, "是否命中？", RED)
    box(d, (1430, 500, 1835, 615), "跳过并继续", "未命中，或重复事件已存在", color=MUTED)
    diamond(d, (1510, 735), 60, "同企业/规则/日期\n事件已存在？", PURPLE)
    route_arrow(d, [(1625, 330), (1625, 450), (240, 450), (240, 515)], ORANGE)
    arrow(d, (380, 580), (455, 580), ORANGE)
    arrow(d, (645, 550), (720, 525), ORANGE, label="阈值型")
    arrow(d, (645, 615), (720, 675), ORANGE, label="变化型")
    arrow(d, (1030, 525), (1130, 565), ORANGE)
    arrow(d, (1030, 675), (1130, 595), ORANGE)
    arrow(d, (1320, 550), (1430, 550), MUTED, label="否")
    arrow(d, (1275, 640), (1450, 700), RED, label="是")
    route_arrow(d, [(1570, 710), (1670, 680), (1670, 615)], MUTED, label="是")

    box(d, (100, 865, 440, 1015), "创建 OPEN 风险事件", "风险等级 / 触发值 / 阈值 / 日期\nevidenceJson 记录指标证据", RED)
    box(d, (515, 865, 850, 1015), "同步 RAG 语料", "event upsert\ncompany aggregate", CYAN)
    box(d, (925, 865, 1260, 1015), "审计留痕", "创建、分派、处理、复核、关闭", PURPLE)
    box(d, (1335, 865, 1900, 1015), "事件处置状态机", "OPEN → ASSIGNED → HANDLED\n→ REVIEWED → CLOSED", GREEN)
    route_arrow(d, [(1450, 770), (1450, 815), (270, 815), (270, 865)], RED, label="否：新事件")
    arrow(d, (440, 940), (515, 940), CYAN)
    arrow(d, (850, 940), (925, 940), PURPLE)
    arrow(d, (1260, 940), (1335, 940), GREEN)
    d.text((82, 1115), "旁路：指标、规则和风险事件发生增删改时，CorpusEvents 同步更新本地检索索引。", font=F_SUB, fill=MUTED)
    save(im, "fig2-risk-rule-flow.png")


def agent_flow():
    im, d = canvas("图3  AI Agent 执行流程", "真实流式输出：工具取证完成后，LangGraph 最终成稿轮逐 token 推送")
    box(d, (70, 215, 310, 315), "用户发起分析", "问题 + 企业 + 深度", BLUE)
    box(d, (370, 215, 650, 315), "Spring 安全边界", "JWT / 权限 / 数据范围", GREEN)
    box(d, (710, 215, 1000, 315), "Python 转发", "HTTP 请求 + SSE 透传", CYAN)
    box(d, (1060, 215, 1325, 315), "输入护栏", "注入/越权/敏感信息检查", RED)
    box(d, (1385, 215, 1925, 315), "规则路由与预算", "意图识别 · quick/standard/deep\n“不要联网”否定识别 · deadline", PURPLE)
    arrow(d, (310, 265), (370, 265), BLUE)
    arrow(d, (650, 265), (710, 265), CYAN)
    arrow(d, (1000, 265), (1060, 265), RED)
    arrow(d, (1325, 265), (1385, 265), PURPLE)

    group(d, (70, 365, 1930, 700), "证据获取与 Agent 编排", CYAN)
    box(d, (105, 455, 350, 575), "上下文记忆", "会话记忆\n跨会话长期记忆", PURPLE)
    box(d, (400, 455, 650, 575), "预取证调度", "工具白名单\n并发与调用预算", GREEN)
    box(d, (700, 405, 990, 515), "内部只读工具", "档案 / 指标 / 事件 / 投诉 / 竞品 / RAG", GREEN)
    box(d, (700, 550, 990, 660), "MCP 联网工具", "按需调用 · 并行搜索\n10 秒硬上限", ORANGE)
    box(d, (1050, 455, 1335, 575), "统一证据池", "来源ID / 网页 URL\n工具轨迹 / 相关度诊断", CYAN)
    box(d, (1395, 435, 1895, 595), "LangGraph 最终成稿轮", "中间工具草稿不外显\n最终轮 token_hook 直连 SSE", PURPLE)
    arrow(d, (350, 515), (400, 515), PURPLE)
    arrow(d, (650, 490), (700, 460), GREEN)
    arrow(d, (650, 540), (700, 605), ORANGE)
    arrow(d, (990, 460), (1050, 500), GREEN)
    arrow(d, (990, 605), (1050, 540), ORANGE)
    arrow(d, (1335, 515), (1395, 515), PURPLE)
    arrow(d, (1780, 575), (1780, 650), BLUE, label="token")
    box(d, (1500, 620, 1890, 680), "浏览器持续显示正文", color=BLUE)

    group(d, (70, 745, 1930, 1125), "成稿校验、持久化与完成事件", RED)
    box(d, (105, 825, 370, 950), "引用核对", "来源编号校验\n必要时自动修补", RED)
    diamond(d, (540, 887), 57, "多 Agent\n复核？", PURPLE)
    box(d, (700, 825, 970, 950), "复核与修订", "严重问题触发回环\n修订后再次核对引用", PURPLE)
    box(d, (1030, 825, 1295, 950), "输出护栏", "结构化回执\n安全与合规检查", RED)
    box(d, (1355, 805, 1615, 970), "落库与留痕", "ai_analysis / trace\n工具耗时 / 证据 / 记忆沉淀", GREEN)
    box(d, (1675, 825, 1895, 950), "done 事件", "最终校验稿覆盖预览稿", BLUE)
    arrow(d, (370, 887), (450, 887), RED)
    arrow(d, (630, 870), (700, 870), PURPLE, label="是")
    arrow(d, (630, 925), (1030, 925), MUTED, label="否")
    arrow(d, (970, 887), (1030, 887), PURPLE)
    arrow(d, (1295, 887), (1355, 887), GREEN)
    arrow(d, (1615, 887), (1675, 887), BLUE)
    box(d, (105, 1010, 600, 1095), "超时 / 熔断 / 模型不可用", "生成本地确定性报告并标记 PARTIAL，不无限等待", ORANGE)
    route_arrow(d, [(600, 1050), (1280, 1050), (1280, 965), (1355, 940)], ORANGE,
                dashed=True, label="降级路径")
    save(im, "fig3-ai-agent-flow.png")


def rag_flow():
    im, d = canvas("图4  RAG 知识库流程", "增量入库与混合召回双链路：业务数据变化即可检索，答案引用可回溯")
    group(d, (70, 190, 1930, 600), "A. 入库链路", GREEN)
    box(d, (105, 270, 390, 410), "数据来源", "文档上传\n企业/指标/投诉/竞品/规则/事件", BLUE)
    box(d, (445, 270, 700, 410), "CorpusEvents", "upsert / remove / aggregate\n3 秒合并窗口", GREEN)
    box(d, (755, 270, 1010, 410), "标准化与切分", "文本抽取 · 分块 · 去重\n企业ID/类型/来源元数据", ORANGE)
    box(d, (1065, 270, 1320, 410), "向量化", "本地 512 维 embedding\n无需外部 Key", PURPLE)
    box(d, (1375, 245, 1640, 435), "双索引持久化", "SQLite 文档表\nFTS5 倒排索引\nNumPy 向量矩阵", CYAN)
    box(d, (1695, 270, 1895, 410), "索引状态", "覆盖率 / pending\n失败项 / 重建", RED)
    arrow(d, (390, 340), (445, 340), BLUE)
    arrow(d, (700, 340), (755, 340), GREEN)
    arrow(d, (1010, 340), (1065, 340), ORANGE)
    arrow(d, (1320, 340), (1375, 340), PURPLE)
    arrow(d, (1640, 340), (1695, 340), CYAN)
    d.text((110, 475), "增量更新", font=F_NODE, fill=GREEN)
    d.text((265, 475), "CRUD 后只更新对应语料；风险事件同时刷新企业聚合文档", font=F_BODY, fill=MUTED)
    d.text((110, 520), "全量重建", font=F_NODE, fill=ORANGE)
    d.text((265, 520), "reindex 扫描全部数据源，重新生成分块、关键词索引和向量索引", font=F_BODY, fill=MUTED)

    group(d, (70, 645, 1930, 1165), "B. 召回与引用链路", CYAN)
    box(d, (105, 735, 355, 860), "用户问题", "企业范围 + topK", BLUE)
    box(d, (405, 735, 655, 860), "查询处理", "关键词抽取\n查询改写 variants", ORANGE)
    box(d, (705, 700, 970, 810), "BM25 召回", "FTS5 关键词匹配", GREEN)
    box(d, (705, 850, 970, 960), "KNN 召回", "本地向量相似度", PURPLE)
    box(d, (1020, 735, 1280, 860), "融合与精排", "路径合并 · 分数归一\nRerank 相关度", CYAN)
    box(d, (1330, 735, 1580, 860), "MMR 去冗余", "多样性选择 + topK\n权限/企业范围过滤", ORANGE)
    box(d, (1630, 735, 1895, 860), "证据卡", "来源ID · 标题 · 分数\n召回路径 · 摘要", RED)
    arrow(d, (355, 797), (405, 797), BLUE)
    arrow(d, (655, 780), (705, 755), GREEN)
    arrow(d, (655, 820), (705, 905), PURPLE)
    arrow(d, (970, 755), (1020, 780), GREEN)
    arrow(d, (970, 905), (1020, 820), PURPLE)
    arrow(d, (1280, 797), (1330, 797), CYAN)
    arrow(d, (1580, 797), (1630, 797), ORANGE)
    box(d, (455, 1010, 800, 1110), "Agent 证据池", "结构化数据与知识库证据合并", CYAN)
    box(d, (890, 1010, 1235, 1110), "模型生成报告", "结论后标注来源ID / [n]", PURPLE)
    box(d, (1325, 1010, 1670, 1110), "Grounding 核对", "悬空引用修补 · 未使用来源分离", RED)
    route_arrow(d, [(1760, 860), (1760, 975), (625, 975), (625, 1010)], RED)
    arrow(d, (800, 1060), (890, 1060), CYAN)
    arrow(d, (1235, 1060), (1325, 1060), PURPLE)
    arrow(d, (1670, 1060), (1880, 1060), GREEN, label="可审计答案")
    save(im, "fig4-rag-knowledge-flow.png")


if __name__ == "__main__":
    architecture()
    risk_flow()
    agent_flow()
    rag_flow()
    print("Generated 4 diagrams in", OUT)
