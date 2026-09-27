from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


OUT = Path(__file__).resolve().parent
W, H = 2400, 3200
BG = "#101214"
NODE = "#082D4A"
NODE_ALT = "#0B3556"
TEXT = "#8ED7FF"
TEXT_STRONG = "#B9E8FF"
MUTED = "#7D8B93"
LINE = "#515B61"
ACCENT = "#2388BE"
OK = "#4EA878"
WARN = "#D89B42"
BAD = "#C86868"

FONT = Path("C:/Windows/Fonts/msyh.ttc")
BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")


def ft(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold and BOLD.exists() else FONT), size)


def make(title, subtitle):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((90, 70), title, font=ft(48, True), fill=TEXT_STRONG)
    d.text((92, 137), subtitle, font=ft(24), fill=MUTED)
    d.line((90, 190, W - 90, 190), fill="#293136", width=2)
    return im, d


def wrapped(d, text, font, width):
    out = []
    for raw in str(text).split("\n"):
        line = ""
        for ch in raw:
            if not line or d.textlength(line + ch, font=font) <= width:
                line += ch
            else:
                out.append(line)
                line = ch
        if line:
            out.append(line)
    return out or [""]


def node(d, cx, cy, w, h, title, body="", fill=NODE, color=TEXT, outline=None):
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    d.rounded_rectangle((x1, y1, x2, y2), radius=12, fill=fill,
                        outline=outline or fill, width=2)
    tf, bf = ft(25, True), ft(20)
    for ts, bs in ((25, 20), (23, 18), (21, 17), (19, 16)):
        tf, bf = ft(ts, True), ft(bs)
        tl = wrapped(d, title, tf, w - 32)
        bl = wrapped(d, body, bf, w - 34) if body else []
        th = len(tl) * (ts + 8) + (10 if bl else 0) + len(bl) * (bs + 7)
        if th <= h - 22:
            break
    y = cy - th / 2
    for line in tl:
        tw = d.textlength(line, font=tf)
        d.text((cx - tw / 2, y), line, font=tf, fill=color)
        y += tf.size + 8
    if bl:
        y += 5
        for line in bl:
            tw = d.textlength(line, font=bf)
            d.text((cx - tw / 2, y), line, font=bf, fill="#8CA5B3")
            y += bf.size + 7
    return (x1, y1, x2, y2)


def decision(d, cx, cy, w, h, text):
    xy = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    d.rounded_rectangle(xy, radius=12, fill=BG, outline=ACCENT, width=2)
    dash = 12
    for x in range(int(xy[0]) + 8, int(xy[2]) - 8, dash * 2):
        d.line((x, xy[1], min(x + dash, xy[2]), xy[1]), fill=TEXT, width=2)
        d.line((x, xy[3], min(x + dash, xy[2]), xy[3]), fill=TEXT, width=2)
    for y in range(int(xy[1]) + 8, int(xy[3]) - 8, dash * 2):
        d.line((xy[0], y, xy[0], min(y + dash, xy[3])), fill=TEXT, width=2)
        d.line((xy[2], y, xy[2], min(y + dash, xy[3])), fill=TEXT, width=2)
    lines = wrapped(d, text, ft(21, True), w - 28)
    yy = cy - len(lines) * 15
    for line in lines:
        tw = d.textlength(line, font=ft(21, True))
        d.text((cx - tw / 2, yy), line, font=ft(21, True), fill=TEXT)
        yy += 30
    return xy


def pill(d, x, y, text, color=TEXT):
    f = ft(16, True)
    tw = d.textlength(text, font=f)
    d.rounded_rectangle((x - tw / 2 - 10, y - 13, x + tw / 2 + 10, y + 14),
                        radius=12, fill=BG, outline=color, width=1)
    d.text((x - tw / 2, y - 10), text, font=f, fill=color)


def edge(d, pts, label=None, color=LINE, width=2, dashed=False):
    for a, b in zip(pts[:-1], pts[1:]):
        if dashed:
            length = max(1, int(math.dist(a, b)))
            for i in range(0, length, 18):
                t1, t2 = i / length, min((i + 9) / length, 1)
                p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
                p2 = (a[0] + (b[0] - a[0]) * t2, a[1] + (b[1] - a[1]) * t2)
                d.line((*p1, *p2), fill=color, width=width)
        else:
            d.line((*a, *b), fill=color, width=width)
    a, b = pts[-2], pts[-1]
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    size = 11
    d.polygon([b,
               (b[0] - size * math.cos(ang - .55), b[1] - size * math.sin(ang - .55)),
               (b[0] - size * math.cos(ang + .55), b[1] - size * math.sin(ang + .55))], fill=color)
    if label:
        longest = max(zip(pts[:-1], pts[1:]), key=lambda p: math.dist(*p))
        mx = (longest[0][0] + longest[1][0]) / 2
        my = (longest[0][1] + longest[1][1]) / 2
        pill(d, mx, my, label, color)


def lane(d, x, y, text):
    d.text((x, y), text, font=ft(22, True), fill="#496270")
    d.line((x, y + 38, W - 100, y + 38), fill="#232A2E", width=1)


def fig1():
    im, d = make("图1  系统总体架构", "深色全链路视图 · 前端、Spring Boot、Python Agent、数据与外部能力")
    node(d, 1200, 280, 360, 92, "用户访问系统")
    node(d, 1200, 430, 330, 92, "Vue 3 前端", "Pinia · Element Plus")
    node(d, 1200, 580, 280, 82, "登录认证")
    node(d, 1200, 730, 390, 96, "Spring Security 权限校验")
    node(d, 1200, 880, 250, 82, "签发 JWT")
    node(d, 1200, 1030, 390, 96, "进入风险管理平台")
    decision(d, 1200, 1190, 300, 82, "选择业务功能")
    for y1, y2 in ((326, 384), (476, 539), (621, 682), (778, 839), (921, 982), (1078, 1149)):
        edge(d, [(1200, y1), (1200, y2)])

    lane(d, 90, 1290, "业务域")
    node(d, 400, 1410, 350, 96, "数据接入中心", "Excel / DOCX / PDF")
    node(d, 830, 1410, 350, 96, "风险规则管理", "阈值与变化率规则")
    node(d, 1570, 1410, 350, 96, "经营分析", "指标 / 投诉 / 竞品")
    node(d, 2000, 1410, 300, 96, "审计追踪")
    edge(d, [(1150, 1190), (400, 1190), (400, 1362)], label="数据导入")
    edge(d, [(1170, 1231), (830, 1231), (830, 1362)], label="规则管理")
    edge(d, [(1230, 1231), (1570, 1231), (1570, 1362)], label="智能分析")
    edge(d, [(1250, 1190), (2000, 1190), (2000, 1362)], label="审计")

    node(d, 400, 1580, 360, 110, "文件校验与结构化", "格式、必填项、数据范围")
    node(d, 830, 1580, 350, 100, "规则写入 MySQL")
    node(d, 1570, 1580, 390, 100, "Spring Boot AI 转发代理")
    node(d, 2000, 1580, 330, 100, "操作审计与 TraceId")
    edge(d, [(400, 1458), (400, 1525)])
    edge(d, [(830, 1458), (830, 1530)])
    edge(d, [(1570, 1458), (1570, 1530)])
    edge(d, [(2000, 1458), (2000, 1530)])

    node(d, 400, 1760, 320, 100, "写入业务表")
    node(d, 830, 1760, 350, 100, "触发风险规则计算")
    node(d, 1570, 1760, 350, 100, "Python FastAPI Agent")
    edge(d, [(400, 1635), (400, 1710)])
    edge(d, [(830, 1630), (830, 1710)])
    edge(d, [(1570, 1630), (1570, 1710)])

    lane(d, 90, 1880, "智能分析与检索域")
    node(d, 610, 2010, 360, 105, "MySQL 业务数据库", "企业 / 指标 / 规则 / 事件")
    node(d, 1200, 2010, 390, 105, "AgentService + LangGraph", "路由、预算、工具编排")
    node(d, 1790, 2010, 340, 105, "本地 RAG 索引", "FTS5 + NumPy 向量")
    edge(d, [(400, 1810), (400, 1860), (610, 1860), (610, 1958)])
    edge(d, [(830, 1810), (830, 1880), (610, 1880), (610, 1958)])
    edge(d, [(1570, 1810), (1570, 1880), (1200, 1880), (1200, 1958)])
    edge(d, [(1200, 2063), (1200, 2140)])
    decision(d, 1200, 2190, 310, 82, "选择取证工具")
    node(d, 520, 2350, 330, 100, "内部结构化工具", "DataScope 约束查询")
    node(d, 1000, 2350, 300, 100, "RAG 检索工具", "混合召回与重排")
    node(d, 1450, 2350, 310, 100, "MCP 网页检索", "并行搜索 · 10 秒上限")
    node(d, 1900, 2350, 300, 100, "外部大模型", "Doubao 推理服务")
    edge(d, [(1100, 2190), (520, 2190), (520, 2300)], label="内部数据")
    edge(d, [(1160, 2231), (1000, 2231), (1000, 2300)], label="知识库")
    edge(d, [(1240, 2231), (1450, 2231), (1450, 2300)], label="外部资料")
    edge(d, [(1300, 2190), (1900, 2190), (1900, 2300)], label="模型推理")
    edge(d, [(520, 2400), (520, 2490), (1200, 2490)])
    edge(d, [(1000, 2400), (1000, 2490), (1200, 2490)])
    edge(d, [(1450, 2400), (1450, 2490), (1200, 2490)])
    edge(d, [(1900, 2400), (1900, 2490), (1200, 2490)])
    node(d, 1200, 2580, 390, 110, "统一证据池与引用核对")
    node(d, 1200, 2760, 390, 105, "SSE 逐 token 输出")
    node(d, 1200, 2930, 390, 100, "结果落库与审计留痕")
    edge(d, [(1200, 2490), (1200, 2525)])
    edge(d, [(1200, 2635), (1200, 2708)])
    edge(d, [(1200, 2813), (1200, 2880)])
    node(d, 1200, 3080, 320, 82, "返回前端展示", fill=NODE_ALT)
    edge(d, [(1200, 2980), (1200, 3039)])
    im.save(OUT / "fig1-system-architecture-dark.png", optimize=True)


def fig2():
    im, d = make("图2  风险规则计算流程", "指标触发、同步降级、阈值与变化率计算、事件去重和处置闭环")
    node(d, 1200, 300, 370, 96, "经营指标进入系统", "Excel 导入 / API 新增")
    node(d, 1200, 470, 320, 90, "指标写入 MySQL")
    decision(d, 1200, 640, 300, 82, "RabbitMQ 可用？")
    node(d, 760, 810, 330, 100, "RabbitMQ 异步派发", "MetricRiskListener")
    node(d, 1640, 810, 330, 100, "同步降级派发", "DirectRiskTaskDispatcher")
    node(d, 1200, 1010, 450, 105, "RiskEngineService.evaluateMetric")
    edge(d, [(1200, 348), (1200, 425)])
    edge(d, [(1200, 515), (1200, 599)])
    edge(d, [(1050, 640), (760, 640), (760, 760)], label="是", color=OK)
    edge(d, [(1350, 640), (1640, 640), (1640, 760)], label="否", color=WARN)
    edge(d, [(760, 860), (760, 920), (1200, 920), (1200, 958)])
    edge(d, [(1640, 860), (1640, 920), (1200, 920), (1200, 958)])
    node(d, 1200, 1190, 360, 100, "查询启用规则", "metric_code 匹配 · enabled=1")
    edge(d, [(1200, 1063), (1200, 1140)])
    decision(d, 1200, 1360, 300, 82, "操作符类型")
    edge(d, [(1200, 1240), (1200, 1319)])
    node(d, 720, 1530, 330, 100, "直接阈值比较", "GT / GE / LT / LE")
    node(d, 1680, 1530, 350, 110, "变化率计算", "读取上一期指标\nCHANGE_GT / CHANGE_LT")
    edge(d, [(1050, 1360), (720, 1360), (720, 1480)], label="阈值型")
    edge(d, [(1350, 1360), (1680, 1360), (1680, 1475)], label="变化型")
    decision(d, 1200, 1750, 280, 82, "规则是否命中？")
    edge(d, [(720, 1580), (720, 1660), (1200, 1660), (1200, 1709)])
    edge(d, [(1680, 1585), (1680, 1660), (1200, 1660), (1200, 1709)])
    node(d, 1850, 1920, 320, 90, "继续下一条规则")
    decision(d, 1200, 1940, 360, 90, "同企业 / 规则 / 日期\n事件已存在？")
    edge(d, [(1340, 1750), (1850, 1750), (1850, 1875)], label="否", color=MUTED)
    edge(d, [(1200, 1791), (1200, 1895)], label="是", color=BAD)
    edge(d, [(1380, 1940), (1850, 1940), (1850, 1965)], label="是：去重")
    node(d, 1200, 2170, 390, 115, "创建 OPEN 风险事件", "风险等级 / 触发值 / 阈值 / 日期\nevidenceJson 指标证据")
    edge(d, [(1200, 1985), (1200, 2112)], label="否：新事件", color=BAD)
    node(d, 700, 2380, 350, 100, "同步 RAG 语料", "event upsert + aggregate")
    node(d, 1700, 2380, 330, 100, "写入审计日志")
    edge(d, [(1100, 2228), (700, 2228), (700, 2330)])
    edge(d, [(1300, 2228), (1700, 2228), (1700, 2330)])
    node(d, 1200, 2580, 390, 100, "风险事件进入处置流程")
    edge(d, [(700, 2430), (700, 2490), (1200, 2490), (1200, 2530)])
    edge(d, [(1700, 2430), (1700, 2490), (1200, 2490), (1200, 2530)])
    for i, (x, label) in enumerate([(420, "OPEN"), (800, "ASSIGNED"), (1200, "HANDLED"),
                                     (1600, "REVIEWED"), (1980, "CLOSED")]):
        node(d, x, 2780, 250, 82, label)
        if i:
            px = [420, 800, 1200, 1600, 1980][i - 1]
            edge(d, [(px + 125, 2780), (x - 125, 2780)])
    edge(d, [(1200, 2630), (1200, 2690), (420, 2690), (420, 2739)])
    node(d, 1200, 2980, 480, 100, "状态变化同步索引并持续审计")
    edge(d, [(1980, 2821), (1980, 2900), (1200, 2900), (1200, 2930)])
    im.save(OUT / "fig2-risk-rule-flow-dark.png", optimize=True)


def fig3():
    im, d = make("图3  AI Agent 执行流程", "LangGraph 工具编排、MCP 限时联网、真实 SSE 流式输出与最终校验")
    node(d, 1200, 290, 380, 96, "用户提交分析问题")
    node(d, 1200, 450, 390, 96, "Spring 鉴权与数据范围检查")
    node(d, 1200, 610, 390, 96, "转发至 Python FastAPI")
    node(d, 1200, 770, 340, 90, "输入护栏检查")
    decision(d, 1200, 930, 330, 82, "输入是否合法？")
    node(d, 1780, 1080, 330, 90, "拒绝请求并记录审计")
    node(d, 1200, 1100, 390, 100, "Agent 路由与预算规划", "quick / standard / deep")
    for a, b in ((338, 402), (498, 562), (658, 725), (815, 889)):
        edge(d, [(1200, a), (1200, b)])
    edge(d, [(1365, 930), (1780, 930), (1780, 1035)], label="否", color=BAD)
    edge(d, [(1200, 971), (1200, 1050)], label="是", color=OK)
    node(d, 1200, 1280, 390, 100, "加载上下文记忆", "会话记忆 + 长期记忆")
    edge(d, [(1200, 1150), (1200, 1230)])
    decision(d, 1200, 1450, 350, 82, "需要哪些证据？")
    edge(d, [(1200, 1330), (1200, 1409)])
    node(d, 430, 1630, 340, 105, "内部结构化工具", "档案 / 指标 / 事件 / 投诉 / 竞品")
    node(d, 1000, 1630, 320, 105, "RAG 检索工具", "BM25 + KNN + Rerank")
    node(d, 1570, 1630, 330, 105, "MCP 网页检索", "按需调用 · 10 秒上限")
    node(d, 2070, 1630, 300, 105, "外部大模型")
    edge(d, [(1050, 1450), (430, 1450), (430, 1578)], label="内部事实")
    edge(d, [(1140, 1491), (1000, 1491), (1000, 1578)], label="知识资料")
    edge(d, [(1260, 1491), (1570, 1491), (1570, 1578)], label="外部对照")
    edge(d, [(1350, 1450), (2070, 1450), (2070, 1578)], label="模型")
    node(d, 1200, 1840, 390, 105, "统一证据池", "来源ID / URL / 工具轨迹 / 相关度")
    for x in (430, 1000, 1570, 2070):
        edge(d, [(x, 1683), (x, 1750), (1200, 1750), (1200, 1788)])
    decision(d, 1200, 2020, 350, 82, "时间预算是否充足？")
    edge(d, [(1200, 1893), (1200, 1979)])
    node(d, 1880, 2180, 360, 100, "本地确定性报告", "标记 PARTIAL，不无限等待")
    node(d, 1200, 2200, 410, 110, "LangGraph 最终成稿轮", "中间工具草稿不外显")
    edge(d, [(1375, 2020), (1880, 2020), (1880, 2130)], label="不足", color=WARN)
    edge(d, [(1200, 2061), (1200, 2145)], label="充足", color=OK)
    node(d, 1200, 2390, 390, 100, "SSE token_hook 实时推送")
    edge(d, [(1200, 2255), (1200, 2340)])
    node(d, 500, 2580, 340, 100, "引用核对与修补")
    decision(d, 1200, 2580, 320, 82, "启用多 Agent 复核？")
    node(d, 1900, 2580, 340, 100, "复核员审查与修订")
    edge(d, [(1200, 2440), (1200, 2539)])
    edge(d, [(1040, 2580), (670, 2580)], label="否")
    edge(d, [(1360, 2580), (1730, 2580)], label="是")
    node(d, 1200, 2780, 390, 100, "输出护栏与结构化回执")
    edge(d, [(500, 2630), (500, 2690), (1200, 2690), (1200, 2730)])
    edge(d, [(1900, 2630), (1900, 2690), (1200, 2690), (1200, 2730)])
    edge(d, [(1880, 2230), (1880, 2690), (1200, 2690)], color=WARN)
    node(d, 820, 2970, 340, 100, "落库与 Trace 留痕")
    node(d, 1580, 2970, 340, 100, "done 最终校验稿")
    edge(d, [(1200, 2830), (1200, 2890), (820, 2890), (820, 2920)])
    edge(d, [(1200, 2890), (1580, 2890), (1580, 2920)])
    im.save(OUT / "fig3-ai-agent-flow-dark.png", optimize=True)


def fig4():
    im, d = make("图4  RAG 知识库流程", "增量入库、双索引持久化、混合召回、MMR 去冗余与引用核对")
    lane(d, 90, 245, "入库链路")
    node(d, 420, 390, 340, 105, "知识文档上传", "PDF / DOCX / TXT")
    node(d, 1200, 390, 380, 105, "结构化业务数据", "企业 / 指标 / 投诉 / 竞品 / 规则 / 事件")
    node(d, 1980, 390, 300, 105, "删除或更新")
    node(d, 1200, 600, 400, 105, "CorpusEvents 事件队列", "upsert / remove / aggregate")
    edge(d, [(420, 443), (420, 520), (1200, 520), (1200, 548)])
    edge(d, [(1200, 443), (1200, 548)])
    edge(d, [(1980, 443), (1980, 520), (1200, 520), (1200, 548)])
    node(d, 1200, 790, 330, 95, "3 秒合并窗口", "批量刷新减少抖动")
    edge(d, [(1200, 653), (1200, 742)])
    node(d, 1200, 980, 390, 105, "文本标准化与分块", "抽取 · 清洗 · 去重 · 来源元数据")
    edge(d, [(1200, 838), (1200, 928)])
    decision(d, 1200, 1150, 330, 82, "索引操作类型")
    edge(d, [(1200, 1033), (1200, 1109)])
    node(d, 570, 1320, 320, 100, "删除旧分块")
    node(d, 1200, 1320, 340, 100, "生成本地 512 维向量")
    node(d, 1830, 1320, 320, 100, "生成关键词文档")
    edge(d, [(1035, 1150), (570, 1150), (570, 1270)], label="remove")
    edge(d, [(1200, 1191), (1200, 1270)], label="vector")
    edge(d, [(1365, 1150), (1830, 1150), (1830, 1270)], label="text")
    node(d, 760, 1510, 340, 100, "NumPy 向量矩阵")
    node(d, 1640, 1510, 360, 100, "SQLite 文档表 + FTS5")
    edge(d, [(1200, 1370), (1200, 1430), (760, 1430), (760, 1460)])
    edge(d, [(1830, 1370), (1830, 1430), (1640, 1430), (1640, 1460)])
    edge(d, [(570, 1370), (570, 1430), (760, 1430)], color=BAD)
    node(d, 1200, 1690, 390, 100, "索引状态与重建管理", "coverage / pending / failed / reindex")
    edge(d, [(760, 1560), (760, 1620), (1200, 1620), (1200, 1640)])
    edge(d, [(1640, 1560), (1640, 1620), (1200, 1620), (1200, 1640)])

    lane(d, 90, 1840, "召回与引用链路")
    node(d, 1200, 1990, 350, 96, "用户问题 + 企业范围")
    node(d, 1200, 2160, 370, 100, "查询处理", "关键词抽取 · 查询改写 variants")
    edge(d, [(1200, 2038), (1200, 2110)])
    node(d, 700, 2350, 320, 100, "BM25 召回", "FTS5 关键词匹配")
    node(d, 1700, 2350, 320, 100, "KNN 召回", "本地向量相似度")
    edge(d, [(1100, 2210), (700, 2210), (700, 2300)])
    edge(d, [(1300, 2210), (1700, 2210), (1700, 2300)])
    node(d, 1200, 2530, 390, 105, "融合与精排", "路径合并 · 分数归一 · Rerank")
    edge(d, [(700, 2400), (700, 2460), (1200, 2460), (1200, 2478)])
    edge(d, [(1700, 2400), (1700, 2460), (1200, 2460), (1200, 2478)])
    node(d, 1200, 2710, 390, 100, "MMR 去冗余 + topK", "权限与企业范围过滤")
    edge(d, [(1200, 2583), (1200, 2660)])
    node(d, 700, 2900, 340, 100, "证据卡", "来源ID / 标题 / 分数 / 摘要")
    node(d, 1200, 2900, 340, 100, "Agent 证据池")
    node(d, 1700, 2900, 340, 100, "Grounding 引用核对")
    edge(d, [(1200, 2760), (1200, 2820), (700, 2820), (700, 2850)])
    edge(d, [(870, 2900), (1030, 2900)])
    edge(d, [(1370, 2900), (1530, 2900)])
    node(d, 1200, 3070, 390, 90, "生成可追溯、可审计答案")
    edge(d, [(1700, 2950), (1700, 3010), (1200, 3010), (1200, 3025)], color=OK)
    im.save(OUT / "fig4-rag-knowledge-flow-dark.png", optimize=True)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("Generated dark flowcharts in", OUT)
