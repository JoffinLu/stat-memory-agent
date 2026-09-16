# -*- coding: utf-8 -*-
"""MEMSCAN · Stat-Memory Agent 控制台（SolScan 风格视觉重构）。

参照 SolScan（Solana 区块链浏览器）的设计语言：
    1. 暗黑底 (#050b09) + 薄荷绿辉光 (#2ce5a7)，边缘 teal 光晕渗出；
    2. 药丸导航条 + 价格胶囊 + 账户胶囊；
    3. Hero 卡：径向 teal 辉光、发光硬币、视觉搜索条、功能药丸、薄荷 CTA；
    4. KV 数据三联卡（Market Overview 式：大数字 + 键值行 + 图标芯片）；
    5. 药丸 Tab（Transfers 式激活态：薄荷绿实底胶囊）；
    6. 交易流水式数据表格（TRANSFER 芯片、蓝色代币徽章、分页行）；
    7. 页脚 "Powered By" + 全宽巨型渐变水印字。
    全站无 serif，纯 Inter 粗排版 + JetBrains Mono 数字。

业务逻辑完全复用既有模块：MemoryManager / AgentExecutor / metrics.ab_test。
启动：
    streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import logging
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st

from app.evaluation.metrics import (
    ABTestReport,
    TaskResult,
    ab_test,
)
from app.memory.manager import MemoryManager
from app.memory.models import MemoryFact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("stat_memory_agent.app")

st.set_page_config(
    page_title="MEMSCAN · 长程记忆智能体控制台",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 1. 视觉系统（SolScan：暗黑 + 薄荷绿辉光 + 纯 sans 粗排版）
# ============================================================

_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --bg:           #050b09;
  --panel:        #0a120f;
  --panel-2:      #0b1411;
  --mint:         #2ce5a7;
  --mint-bright:  #3ef0b4;
  --mint-deep:    #0f7a58;
  --blue:         #4d8dff;
  --text:         #f2f7f5;
  --muted:        #8fa39b;
  --faint:        #5c6f67;
  --line:         rgba(255,255,255,.06);
  --mint-line:    rgba(44,229,167,.16);
  --mono: 'JetBrains Mono', Consolas, monospace;
  --sans: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

html, body, .stApp {
  background:
    radial-gradient(900px 480px at 6% -6%, rgba(19,150,108,.38), transparent 58%),
    radial-gradient(760px 520px at 108% 42%, rgba(15,110,80,.16), transparent 60%),
    radial-gradient(1100px 460px at 40% 118%, rgba(14,96,70,.12), transparent 62%),
    var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--sans);
}
h1,h2,h3 { font-family: var(--sans) !important; color: var(--text) !important; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.4rem !important; max-width: 1480px !important; padding-bottom: 0 !important; }
a { color: var(--mint) !important; }

/* ---------- 导航条 ---------- */
.nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 22px; margin-bottom: 16px;
  background: rgba(9,16,13,.78);
  border: 1px solid var(--line);
  border-radius: 16px;
}
.nav-brand { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 1.18rem; letter-spacing: .04em; color: var(--text); }
.nav-brand .mark {
  width: 26px; height: 26px; border-radius: 50%;
  background: radial-gradient(circle at 32% 28%, #b9ffe4, var(--mint) 42%, #0a4a37 88%);
  box-shadow: 0 0 18px rgba(44,229,167,.55);
}
.nav-links { display: flex; gap: 26px; font-size: .8rem; color: var(--muted); font-weight: 500; }
.nav-links b { color: var(--text); font-weight: 600; }
.nav-right { display: flex; align-items: center; gap: 10px; }
.pill {
  display: inline-flex; align-items: center; gap: 7px;
  font-family: var(--mono); font-size: .72rem; font-weight: 600;
  padding: 7px 14px; border-radius: 999px;
  background: var(--panel-2); border: 1px solid var(--line);
  color: var(--muted);
}
.pill .up { color: var(--mint); }
.pill .dot { width: 6px; height: 6px; border-radius: 50%; animation: pulse 2.2s ease-in-out infinite; }
.pill.demo .dot { background: #f5b83d; box-shadow: 0 0 8px #f5b83d; }
.pill.live .dot { background: var(--mint); box-shadow: 0 0 8px var(--mint); }
.pill.account { color: var(--text); gap: 8px; }
.pill.account .av { width: 18px; height: 18px; border-radius: 50%; background: linear-gradient(135deg, #2ce5a7, #0f7a58); }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.45;transform:scale(1.25)} }

/* ---------- Hero 卡 ---------- */
.hero {
  position: relative; overflow: hidden;
  border-radius: 24px;
  border: 1px solid rgba(44,229,167,.14);
  background:
    radial-gradient(680px 330px at 44% 74%, rgba(23,168,120,.42), rgba(12,42,32,.20) 46%, transparent 72%),
    linear-gradient(180deg, #0a1411 0%, #081009 100%);
  padding: 40px 44px 46px 44px;
  margin-bottom: 16px;
}
.hero-grid { display: flex; align-items: center; gap: 24px; position: relative; z-index: 2; }
.hero-left { flex: 1.25; min-width: 0; }
.hero-eyebrow { font-family: var(--mono); font-size: .68rem; letter-spacing: .22em; color: var(--mint); margin-bottom: 10px; font-weight: 600; }
.hero-title { font-size: clamp(1.9rem, 3.2vw, 2.7rem); font-weight: 800; letter-spacing: -.02em; margin-bottom: 22px; }
/* 视觉搜索条（SolScan 同款：暗底胶囊 + 薄荷圆钮） */
.hero-search {
  display: flex; align-items: center; justify-content: space-between;
  max-width: 430px; padding: 8px 8px 8px 20px;
  background: rgba(4,9,7,.72); border: 1px solid rgba(255,255,255,.08);
  border-radius: 999px; backdrop-filter: blur(8px);
}
.hero-search .ph { font-size: .78rem; color: var(--faint); }
.hero-search .go {
  width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
  display: inline-flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--mint-bright), #17bd88);
  color: #04120c; font-size: .85rem; font-weight: 800;
  box-shadow: 0 0 22px rgba(44,229,167,.45);
}
/* 发光硬币 */
.hero-coin { flex: .8; display: flex; justify-content: center; }
.coin {
  width: 132px; height: 132px; border-radius: 50%;
  background: conic-gradient(from 210deg, #c8ffe9, var(--mint) 24%, #0a4a37 52%, #17bd88 74%, #c8ffe9);
  box-shadow: 0 0 90px rgba(44,229,167,.42), 0 18px 50px rgba(0,0,0,.5), inset 0 0 34px rgba(255,255,255,.22);
  display: flex; align-items: center; justify-content: center;
  animation: float 6s ease-in-out infinite;
}
.coin-face {
  width: 96px; height: 96px; border-radius: 50%;
  border: 2px solid rgba(255,255,255,.5);
  background: radial-gradient(circle at 34% 28%, rgba(255,255,255,.35), transparent 46%), linear-gradient(160deg, #17bd88, #0a4a37);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 2.4rem; font-weight: 700; color: #eafff5;
  text-shadow: 0 0 18px rgba(255,255,255,.55);
}
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-12px)} }
.hero-right { flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 12px; }
.fpill-row { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.fpill {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: .72rem; font-weight: 600; color: var(--muted);
  padding: 7px 13px; border-radius: 999px;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
}
.fpill .ic { color: var(--mint); font-size: .68rem; }
.cta {
  margin-top: 18px; padding: 11px 22px; border-radius: 12px;
  background: linear-gradient(135deg, var(--mint-bright), #17bd88);
  color: #04120c; font-weight: 700; font-size: .84rem;
  box-shadow: 0 6px 30px rgba(44,229,167,.35);
}
.cta-note { font-family: var(--mono); font-size: .64rem; color: var(--faint); letter-spacing: .06em; }

/* ---------- KV 三联卡 ---------- */
.trio { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 18px; }
.kvcard { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 20px 22px; }
.kvcard .t { font-size: .92rem; font-weight: 700; margin-bottom: 16px; }
.bignums { display: flex; gap: 34px; margin-bottom: 16px; flex-wrap: wrap; }
.bignums .n { font-family: var(--mono); font-size: 1.55rem; font-weight: 700; color: var(--text); letter-spacing: -.02em; }
.bignums .l { font-size: .68rem; color: var(--faint); margin-top: 3px; }
.kvrow { display: flex; align-items: center; justify-content: space-between; padding: 7px 0; border-top: 1px solid rgba(255,255,255,.04); }
.kvrow .k { font-size: .78rem; color: var(--muted); display: flex; align-items: center; gap: 8px; }
.kvrow .k .ic {
  width: 20px; height: 20px; border-radius: 50%; flex-shrink: 0;
  display: inline-flex; align-items: center; justify-content: center;
  background: rgba(44,229,167,.12); border: 1px solid rgba(44,229,167,.28);
  color: var(--mint); font-size: .58rem;
}
.kvrow .v { font-size: .76rem; color: var(--text); font-weight: 500; }
.kvrow .v .chip-inline {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: var(--mono); font-size: .66rem; font-weight: 600;
  padding: 4px 10px; border-radius: 999px;
  background: rgba(44,229,167,.08); border: 1px solid rgba(44,229,167,.22); color: var(--mint);
}

/* ---------- 药丸导航（st.radio 横向改造，保证点击可靠性） ---------- */
div[data-testid="stRadio"] { margin-bottom: 14px; }
div[data-testid="stRadio"] [role="radiogroup"] {
  display: inline-flex; gap: 6px; flex-wrap: wrap;
  background: rgba(255,255,255,.03);
  border: 1px solid var(--line); border-radius: 999px; padding: 5px 6px;
  width: fit-content;
}
div[data-testid="stRadio"] [role="radiogroup"] label {
  display: inline-flex; align-items: center; justify-content: center;
  background: transparent; border: none; border-radius: 999px;
  padding: 7px 20px; margin: 0; cursor: pointer;
  transition: background .2s ease;
}
div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child { display: none; }
div[data-testid="stRadio"] [role="radiogroup"] label p {
  font-size: .8rem; font-weight: 600; color: var(--muted) !important; margin: 0;
}
div[data-testid="stRadio"] [role="radiogroup"] label:hover p { color: var(--text) !important; }
div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"],
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(135deg, var(--mint-bright), #17bd88);
}
div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] p,
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {
  color: #04120c !important; font-weight: 700;
}

/* ---------- 流水表格 ---------- */
.ledger-wrap { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; overflow: hidden; }
table.ledger { width: 100%; border-collapse: collapse; }
table.ledger th {
  font-family: var(--mono); font-size: .62rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--faint); text-align: left; font-weight: 600;
  padding: 13px 16px; border-bottom: 1px solid var(--line);
}
table.ledger td { padding: 13px 16px; border-bottom: 1px solid rgba(255,255,255,.04); font-size: .8rem; color: var(--text); vertical-align: middle; }
table.ledger tbody tr:last-child td { border-bottom: none; }
table.ledger tbody tr { transition: background .15s ease; }
table.ledger tbody tr:hover { background: rgba(44,229,167,.035); }
td.t-time { font-family: var(--mono); font-size: .72rem; color: var(--muted); white-space: nowrap; }
.act {
  display: inline-block; font-family: var(--mono); font-size: .58rem; font-weight: 700;
  letter-spacing: .1em; padding: 3px 9px; border-radius: 5px;
  background: rgba(44,229,167,.10); color: var(--mint); border: 1px solid rgba(44,229,167,.25);
}
.act.fuse  { background: rgba(77,141,255,.10); color: var(--blue); border-color: rgba(77,141,255,.28); }
.act.user  { background: rgba(245,184,61,.10); color: #f5b83d; border-color: rgba(245,184,61,.28); }
td.c-content { max-width: 430px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
td.v-conf { font-family: var(--mono); font-weight: 700; color: var(--mint); white-space: nowrap; }
td.v-conf .bar { display: inline-block; width: 52px; height: 3px; border-radius: 99px; background: rgba(44,229,167,.14); margin-left: 10px; vertical-align: middle; position: relative; }
td.v-conf .bar i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; background: linear-gradient(90deg, #17bd88, var(--mint-bright)); }
.tok {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--mono); font-size: .68rem; font-weight: 700;
}
.tok .c { width: 18px; height: 18px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: .52rem; color: #fff; }
.tok.mem .c { background: linear-gradient(135deg, #4d8dff, #2f5fd0); }
.tok.mnt .c { background: linear-gradient(135deg, var(--mint-bright), #17bd88); color: #04120c; }
.pagebar { display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; }
.pagebar .l { font-size: .74rem; color: var(--muted); display: flex; align-items: center; gap: 8px; }
.pagebar .l b { color: var(--text); }
.pagebar .pg { display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: .7rem; color: var(--muted); }
.pagebar .pg .cur {
  min-width: 26px; height: 26px; padding: 0 6px; border-radius: 8px;
  display: inline-flex; align-items: center; justify-content: center;
  background: rgba(44,229,167,.14); border: 1px solid rgba(44,229,167,.35); color: var(--mint); font-weight: 700;
}
.toolrow { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.mint-tag {
  display: inline-flex; align-items: center; padding: 7px 16px; border-radius: 999px;
  background: linear-gradient(135deg, var(--mint-bright), #17bd88);
  color: #04120c; font-size: .74rem; font-weight: 700;
}

/* ---------- 对话 ---------- */
div[data-testid="stChatMessage"] {
  background: rgba(10,18,15,.72) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  padding: 10px 16px !important; margin-bottom: 10px !important;
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { color: var(--text); font-size: .9rem; line-height: 1.6; }
div[data-testid="stChatInput"] textarea {
  background: rgba(4,9,7,.72) !important; border: 1px solid rgba(255,255,255,.09) !important;
  border-radius: 999px !important; color: var(--text) !important; font-size: .9rem;
  caret-color: var(--mint);
}
div[data-testid="stChatInput"] textarea:focus {
  border-color: rgba(44,229,167,.45) !important;
  box-shadow: 0 0 0 3px rgba(44,229,167,.10), 0 0 24px rgba(44,229,167,.12) !important;
}
div[data-testid="stChatInput"] footer button {
  background: linear-gradient(135deg, var(--mint-bright), #17bd88) !important;
  border-radius: 50% !important; border: none !important;
}
div[data-testid="stChatInput"] footer button svg { fill: #04120c !important; color: #04120c !important; }

/* ---------- 徽章 / 芯片 ---------- */
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--mono); font-size: .64rem; font-weight: 600; letter-spacing: .08em;
  padding: 4px 11px; border-radius: 999px; margin: 0 6px 6px 0;
  background: rgba(44,229,167,.08); color: var(--mint); border: 1px solid rgba(44,229,167,.22);
}
.chip.amber { background: rgba(245,184,61,.08); color: #f5b83d; border-color: rgba(245,184,61,.26); }
.chip.blue  { background: rgba(77,141,255,.08); color: var(--blue); border-color: rgba(77,141,255,.26); }
.chip.rose  { background: rgba(251,113,133,.08); color: #fb7185; border-color: rgba(251,113,133,.26); }
.chip.dim   { background: transparent; color: var(--faint); border-color: rgba(255,255,255,.08); }

/* ---------- 控件 ---------- */
div.stButton > button, div[data-testid="stDownloadButton"] > button {
  background: rgba(255,255,255,.04) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  border-radius: 999px !important;
  font-family: var(--mono) !important; font-size: .72rem !important; font-weight: 600;
  padding: .48rem 1.1rem !important;
  transition: all .2s ease;
}
div.stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
  border-color: rgba(44,229,167,.45) !important; color: var(--mint) !important;
}
div.stButton > button[kind="primary"], div[data-testid="stDownloadButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, var(--mint-bright), #17bd88) !important;
  color: #04120c !important; border: none !important;
  box-shadow: 0 4px 22px rgba(44,229,167,.30);
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: #04120c !important; }
div[data-testid="stJson"] {
  background: rgba(4,9,7,.7) !important; border: 1px solid var(--line) !important;
  border-radius: 14px !important; font-family: var(--mono) !important; font-size: .74rem !important;
  max-height: 320px; overflow-y: auto;
}
div[data-testid="stExpander"] {
  background: rgba(10,18,15,.55) !important;
  border: 1px solid var(--line) !important; border-radius: 14px !important;
}
div[data-testid="stExpander"] summary { font-size: .8rem; color: var(--muted); }
div[data-testid="stSlider"] label, div[data-testid="stCheckbox"] label { color: var(--muted) !important; font-size: .76rem; }
div[data-testid="stSelectbox"] > div > div { background: var(--panel-2); border-radius: 10px; }
div[data-testid="stTextInput"] input {
  background: rgba(4,9,7,.72) !important; border: 1px solid rgba(255,255,255,.09) !important;
  border-radius: 999px !important; color: var(--text) !important;
}
div[data-testid="stTextInput"] input:focus { border-color: rgba(44,229,167,.45) !important; }
div[data-testid="stAlert"] {
  background: rgba(44,229,167,.05) !important; border: 1px solid rgba(44,229,167,.16) !important;
  border-radius: 12px !important; color: var(--text) !important;
}
img { border-radius: 14px; }
.stDataFrame { border-radius: 14px !important; border: 1px solid var(--line) !important; }

/* ---------- 滚动条 ---------- */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(44,229,167,.18); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(44,229,167,.4); }

/* ---------- 页脚：Powered By + 巨型水印 ---------- */
.powered { display: flex; justify-content: space-between; align-items: flex-start; gap: 40px; padding: 64px 8px 8px 8px; }
.powered .pl .small { font-size: .82rem; color: var(--muted); font-weight: 500; }
.powered .pl .big { font-size: 1.6rem; font-weight: 800; letter-spacing: -.01em; margin-top: 4px; }
.powered .pr { max-width: 420px; font-size: .74rem; color: var(--muted); line-height: 1.7; text-align: right; }
.wm {
  font-size: clamp(72px, 13.5vw, 208px); font-weight: 800; line-height: .92;
  letter-spacing: -.025em; white-space: nowrap; user-select: none;
  background: linear-gradient(96deg, #ecf8f2 0%, #d7efe4 58%, rgba(236,248,242,.05) 97%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  margin: 8px 0 0 -6px;
}
.foot-links {
  display: flex; justify-content: space-between; align-items: center;
  border-top: 1px solid var(--line); margin-top: 26px; padding: 20px 8px 26px 8px;
  font-size: .74rem; color: var(--muted);
}
.foot-links .grp { display: flex; gap: 30px; }
.foot-links .cp { font-family: var(--mono); font-size: .66rem; color: var(--faint); }
</style>
"""


def inject_visual_layer() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ============================================================
# 2. 演示后端：LLM 替身 + 嵌入
# ============================================================

class _DeterministicEmbedder:
    """零依赖字符哈希嵌入（demo 用，确定性）。"""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def __call__(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for i, ch in enumerate(text):
            vec[ord(ch) % self.dim] += 1.0 / (1 + i * 0.01)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class StreamlitLLM:
    """演示 LLM：按 system prompt 分发到规划 / ReAct / Critic 三个契约分支。"""

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self.last_messages: Optional[List[Any]] = None
        self.invocation_count = 0
        # 故障注入：>0 时 Critic 连续拒绝（演示自纠错用，每拒绝一次减一）
        self._pending_failures = 0

    @staticmethod
    def _msg_content(m: Any) -> str:
        return m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")

    @staticmethod
    def _msg_role(m: Any) -> str:
        return m.get("role", "") if isinstance(m, dict) else getattr(m, "type", "")

    def invoke(self, messages: List[Any]) -> SimpleNamespace:
        self.invocation_count += 1
        self.last_messages = messages

        system_text = ""
        user_text = ""
        for m in messages:
            if self._msg_role(m) == "system":
                system_text = self._msg_content(m)
            elif self._msg_role(m) == "user":
                user_text = self._msg_content(m)

        if "subtasks" in system_text and "depends_on" in system_text:
            return SimpleNamespace(content=self._plan_json(user_text))
        if "passed" in system_text and "reason" in system_text:
            if self._pending_failures > 0:
                self._pending_failures -= 1
                return SimpleNamespace(
                    content=(
                        '{"passed": false, "reason": '
                        '"注入的演示故障：输出未覆盖子任务要求的交付物"}'
                    )
                )
            return SimpleNamespace(
                content='{"passed": true, "reason": "符合子任务契约（演示裁决）"}'
            )
        return SimpleNamespace(content=self._react_answer(user_text))

    def _plan_json(self, user_text: str) -> str:
        import json

        keywords = [w for w in user_text.split() if 2 <= len(w) <= 12][:3] or [
            "背景调研", "方案设计", "验证交付",
        ]
        n = self._rng.choice([2, 3])
        items = [
            {
                "id": f"s{i + 1}",
                "description": f"子任务：{keywords[i % len(keywords)]}（步骤 {i + 1}/{n}）",
                "depends_on": [f"s{j + 1}" for j in range(i)] if i else [],
            }
            for i in range(n)
        ]
        return json.dumps({"subtasks": items}, ensure_ascii=False)

    @staticmethod
    def _react_answer(user_text: str) -> str:
        snippet = user_text.strip().split("\n")[-1]
        snippet = snippet[:77] + "..." if len(snippet) > 80 else snippet
        return (
            "Thought: 已基于记忆上下文完成推理。\n"
            f"Final Answer: 子任务「{snippet}」已执行并通过验证（演示模式）。"
        )


# ============================================================
# 3. 会话状态
# ============================================================

def _init_state() -> None:
    if "_inited" in st.session_state:
        return

    from app.core.llm_client import get_llm_client, ping_vllm

    real_llm = get_llm_client()
    vllm_alive = bool(real_llm) and ping_vllm("http://localhost:8000/v1", timeout=0.6)
    demo_mode = real_llm is None or not vllm_alive
    llm_client: Any = StreamlitLLM(seed=7) if demo_mode else real_llm

    from app.retrieval.hybrid_search import InMemoryVectorStore

    embedding_fn = _DeterministicEmbedder(dim=128)
    vector_store = InMemoryVectorStore(embedding_fn=embedding_fn)

    memory = MemoryManager(
        llm_client=llm_client,
        vector_store=vector_store,
        embedding_fn=embedding_fn,
        k_rrf=60,
        statistical=True,
    )
    from app.agent.planner import AgentExecutor
    from app.agent.runtime import AgentRuntime

    executor = AgentExecutor(executor_llm=llm_client, critic_llm=llm_client)
    runtime = AgentRuntime(memory=memory, executor=executor)

    st.session_state.update({
        "_inited": True,
        "_demo_mode": demo_mode,
        "_llm": llm_client,
        "_memory": memory,
        "_executor": executor,
        "_runtime": runtime,
        "messages": [],
        "ab_results": None,
        "ab_n": 20,
        "runs": 0,
        "ok_runs": 0,
        "ledger": [],  # 流水：{"time","action","content","conf"}
    })

    _seed_demo_memories(memory)


def _seed_demo_memories(memory: MemoryManager) -> None:
    samples = [
        ("用户偏好 pytest 而非 unittest 编写测试", 0.85),
        ("项目截止日期：2026 年 10 月 31 日", 0.92),
        ("向量库采用 ChromaDB 本地持久化", 0.70),
        ("记忆压缩触发阈值：5 条", 0.65),
        ("冲突消解使用时间衰减（λ=0.01）", 0.78),
        ("A/B 检验使用 Welch's t 检验", 0.88),
        ("微调采用 QLoRA + DPO，base = Qwen2.5-7B", 0.80),
    ]
    for content, conf in samples:
        memory.add_fact(MemoryFact(content=content, confidence=conf))
        _append_ledger("EXTRACT", content, conf)


def _append_ledger(action: str, content: str, conf: float) -> None:
    st.session_state["ledger"].insert(0, {
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "action": action,
        "content": content,
        "conf": round(float(conf), 2),
    })


# ============================================================
# 4. 业务动作
# ============================================================

def ingest_chat_turn(memory: MemoryManager, user_text: str) -> Tuple[str, float]:
    import uuid

    fact = MemoryFact(
        id=str(uuid.uuid4()),
        content=user_text.strip(),
        confidence=round(random.uniform(0.55, 0.95), 2),
        timestamp=datetime.now(timezone.utc),
    )
    memory.add_fact(fact)
    _append_ledger("USER", user_text.strip(), fact.confidence)
    return "stored", fact.confidence


def run_agent(executor, task: str) -> Dict[str, Any]:
    try:
        report = executor.run(task, memory_context="")
        return {
            "status": report.status,
            "results": [
                {
                    "subtask_id": r.subtask_id,
                    "description": r.description,
                    "answer": (r.answer or "")[:200],
                    "passed": r.passed,
                    "retries": r.retries,
                }
                for r in report.results
            ],
            "transitions": [f"{s}→{d}" for s, d in report.transitions],
            "error": report.error,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("AgentExecutor 执行失败")
        return {"status": "failed", "results": [], "transitions": [], "error": str(exc)}


def _generate_demo_ab_results(n: int = 20, seed: int = 42) -> ABTestReport:
    """生成方向与真实 benchmark 一致的 A/B 演示数据（B 组显著优于 A）。"""
    rng = np.random.default_rng(seed)

    a = (
        np.clip(rng.normal(0.55, 0.10, n), 0, 1),
        np.clip(rng.normal(0.50, 0.12, n), 0, 1),
        np.clip(rng.normal(0.60, 0.10, n), 0, 1),
        (rng.random(n) < 0.65).astype(float),
        rng.normal(7.5, 1.2, n).clip(min=2),
        rng.normal(4200, 600, n).clip(min=500),
    )
    b = (
        np.clip(rng.normal(0.82, 0.07, n), 0, 1),
        np.clip(rng.normal(0.78, 0.08, n), 0, 1),
        np.clip(rng.normal(0.85, 0.06, n), 0, 1),
        (rng.random(n) < 0.93).astype(float),
        rng.normal(5.2, 1.0, n).clip(min=2),
        rng.normal(3300, 480, n).clip(min=500),
    )

    def to_results(arrays, tag):
        recall, mrr, ndcg, success, steps, tokens = arrays
        out = []
        for i in range(n):
            relevant = [f"mem_{tag}_{i}_a", f"mem_{tag}_{i}_b"]
            k = int(round(recall[i] * len(relevant)))
            out.append(TaskResult(
                task_id=f"{tag}_{i:02d}",
                relevant_ids=relevant,
                retrieved_ids=relevant[:k] or [f"mem_unrel_{tag}_{i}"],
                success=bool(success[i] > 0.5),
                n_steps=int(round(steps[i])),
                tokens=int(round(tokens[i])),
            ))
        return out

    output_dir = str(PROJECT_ROOT / "evaluation")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return ab_test(
        group_a_results=to_results(a, "a"),
        group_b_results=to_results(b, "b"),
        k=5, alpha=0.05, output_dir=output_dir,
        make_chart=True, make_report=True,
    )


def _render_dark_chart(report: ABTestReport, output_path: str) -> str:
    """深色主题版 A/B 对比图（数据来自 metrics.ab_test，视觉与控制台统一）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    from app.evaluation.metrics import _group_ci_halfwidth  # 复用 CI 半宽计算

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8))
    fig.patch.set_facecolor("#081009")
    color_a, color_b = "#3d5a4c", "#2ce5a7"

    fig.suptitle(
        f"A/B 消融  ·  基础 Agent (n={report.n_a})  vs  统计优化 Agent (n={report.n_b})"
        f"   —   Welch's t-test, α={report.alpha}",
        fontsize=12, color="#f2f7f5", fontweight="bold", y=0.985,
    )

    for ax, comp in zip(axes.flat, report.comparisons):
        ax.set_facecolor("#0a1512")
        yerr = [
            _group_ci_halfwidth(report.arrays_a[comp.metric]),
            _group_ci_halfwidth(report.arrays_b[comp.metric]),
        ]
        bars = ax.bar(
            ["A 基础", "B 统计"], [comp.mean_a, comp.mean_b],
            color=[color_a, color_b], width=0.52,
            yerr=yerr, capsize=4,
            error_kw={"ecolor": "#5c6f67", "elinewidth": 1},
        )
        for idx, (bar, value) in enumerate(zip(bars, (comp.mean_a, comp.mean_b))):
            ax.text(bar.get_x() + bar.get_width() / 2, value + yerr[idx],
                    f"{value:.3f}", ha="center", va="bottom", fontsize=8.5, color="#f2f7f5")
        star = "  *" if comp.significant else ""
        note = " ↓优" if not comp.higher_better else ""
        ax.set_title(f"{comp.label}{note}   p={comp.p_value:.4f}{star}",
                     fontsize=10, color="#8fa39b")
        ax.tick_params(colors="#8fa39b", labelsize=9)
        ax.grid(axis="y", alpha=0.15, color="#2ce5a7", linestyle="--")
        for s in ax.spines.values():
            s.set_color((44 / 255, 229 / 255, 167 / 255, 0.25))
        ax.margins(y=0.28)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(out)


# ============================================================
# 5. 渲染组件（SolScan 布局）
# ============================================================

def render_nav(demo_mode: bool) -> None:
    facts = list(st.session_state["_memory"]._searcher.facts)
    avg_conf = sum(f.confidence for f in facts) / len(facts) if facts else 0.0
    status = (
        '<div class="pill demo"><span class="dot"></span>演示模式</div>'
        if demo_mode else
        '<div class="pill live"><span class="dot"></span>vLLM 在线</div>'
    )
    st.markdown(
        f'<div class="nav">'
        f'<div class="nav-brand"><span class="mark"></span>MEMSCAN</div>'
        f'<div class="nav-links"><b>台账</b><span>控制台</span><span>检索</span><span>基准测试</span><span>部署</span></div>'
        f'<div class="nav-right">'
        f'<div class="pill"><span>置信度</span><span class="up">{avg_conf:.2f}</span></div>'
        f'{status}'
        f'<div class="pill account"><span class="av"></span>管理员 ▾</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_hero(demo_mode: bool) -> None:
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-grid">
    <div class="hero-left">
      <div class="hero-eyebrow">统计驱动 · 长程记忆智能体</div>
      <div class="hero-title">长程记忆引擎</div>
      <div class="hero-search">
        <span class="ph">任务输入框已固定在页面底部 ↓ 直接输入即可</span>
        <span class="go">↓</span>
      </div>
    </div>
    <div class="hero-coin"><div class="coin"><div class="coin-face">Σ</div></div></div>
    <div class="hero-right">
      <div class="fpill-row">
        <span class="fpill"><span class="ic">◉</span> 召回</span>
        <span class="fpill"><span class="ic">⇄</span> 冲突消解</span>
        <span class="fpill"><span class="ic">▦</span> 压缩</span>
        <span class="fpill"><span class="ic">✓</span> 验证</span>
      </div>
      <div class="cta">一键运行 A/B 基准测试</div>
      <div class="cta-note">冲突消解 → 序贯检验 → 过程监控 · 全流程统计驱动</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if demo_mode:
        st.markdown(
            '<span class="chip amber">离线模式 · StreamlitLLM 演示后端</span>'
            '<span class="chip dim">配置 SMA_LLM_API_KEY 并启动本地 vLLM，重启后即可切换为在线模式</span>',
            unsafe_allow_html=True,
        )


def _kvrow(icon: str, key: str, value: str, chip: bool = False) -> str:
    v = (
        f'<span class="chip-inline">{value}</span>' if chip
        else f'<span>{value}</span>'
    )
    return (
        f'<div class="kvrow"><div class="k"><span class="ic">{icon}</span>{key}</div>'
        f'<div class="v">{v}</div></div>'
    )


def render_trio(memory: MemoryManager) -> None:
    facts = list(memory._searcher.facts)
    n = len(facts)
    avg_conf = sum(f.confidence for f in facts) / n if n else 0.0
    archive_n = memory.archive_size
    vs_name = type(memory._searcher._vector_store).__name__
    ab: Optional[ABTestReport] = st.session_state.get("ab_results")
    wins = len(ab.significant_improvements()) if ab else 0
    runs = st.session_state.get("runs", 0)
    ok_runs = st.session_state.get("ok_runs", 0)
    success_rate = (ok_runs / runs) if runs else 0.0
    demo = st.session_state.get("_demo_mode", True)

    st.markdown(
        f"""
<div class="trio">
  <div class="kvcard">
    <div class="t">记忆总览</div>
    <div class="bignums">
      <div><div class="n">{n}</div><div class="l">记忆总数</div></div>
      <div><div class="n" style="color:var(--mint)">{avg_conf:.2f}</div><div class="l">平均置信度</div></div>
      <div><div class="n">{archive_n}</div><div class="l">已归档</div></div>
    </div>
    {_kvrow('◈', '向量库', vs_name)}
    {_kvrow('⇌', '混合检索', 'BM25 + 向量 + RRF')}
    {_kvrow('▦', '记忆压缩', 'TF-IDF ≥ 0.85', chip=True)}
  </div>
  <div class="kvcard">
    <div class="t">执行概况</div>
    <div class="bignums">
      <div><div class="n">{runs}</div><div class="l">执行次数</div></div>
      <div><div class="n" style="color:var(--mint)">{success_rate * 100:.0f}%</div><div class="l">成功率</div></div>
    </div>
    {_kvrow('⟐', '状态机', '六状态流转')}
    {_kvrow('✓', 'Critic 校验', 'LLM 验证')}
    {_kvrow('↺', '重试停止规则', 'SPRT α=.05 β=.10', chip=True)}
  </div>
  <div class="kvcard">
    <div class="t">运行时</div>
    {_kvrow('⚡', 'LLM 后端', 'vLLM · Qwen2.5-7B' if not demo else 'StreamlitLLM（演示）')}
    {_kvrow('◇', '嵌入模型', '哈希 128 维（离线）')}
    {_kvrow('λ', '时间衰减', 'λ = 0.01 / 天')}
    {_kvrow('◎', 'SPC 监控', '3σ · 窗口 30', chip=True)}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _ledger_table(rows: List[Dict[str, Any]]) -> str:
    act_label = {"EXTRACT": "抽取", "FUSE": "融合", "USER": "用户", "RRF": "检索", "RETRY": "重试"}
    trs = []
    for r in rows:
        cls = {"EXTRACT": "", "FUSE": "fuse", "USER": "user", "RETRY": "fuse"}.get(r["action"], "")
        label = act_label.get(r["action"], r["action"])
        pct = int(round(r["conf"] * 100))
        trs.append(
            f'<tr>'
            f'<td class="t-time">{r["time"]}</td>'
            f'<td><span class="act {cls}">{label}</span></td>'
            f'<td class="c-content" title="{r["content"]}">{r["content"]}</td>'
            f'<td class="v-conf">{r["conf"]:.2f}<span class="bar"><i style="width:{pct}%"></i></span></td>'
            f'<td><span class="tok mem"><span class="c">Σ</span>记忆</span></td>'
            f'</tr>'
        )
    return (
        '<div class="ledger-wrap"><table class="ledger">'
        '<thead><tr><th>时间</th><th>动作</th><th>内容</th><th>置信度</th><th>来源</th></tr></thead>'
        f'<tbody>{"".join(trs)}</tbody></table>'
        f'<div class="pagebar">'
        f'<div class="l">共找到 <b>{len(rows)}</b> 条记忆</div>'
        f'<div class="pg">第 1 页 &nbsp; <span class="cur">1</span> &nbsp; ›</div>'
        f'</div></div>'
    )


def tab_ledger(memory: MemoryManager) -> None:
    facts = sorted(memory._searcher.facts, key=lambda f: -f.confidence)
    ledger = st.session_state.get("ledger", [])

    tool = st.columns([1, 1, 4])
    with tool[0]:
        high_only = st.checkbox("仅高置信 ≥ 0.80", value=False)
    with tool[1]:
        st.download_button(
            "导出 CSV",
            data=_memory_csv(facts),
            file_name="memory_ledger.csv",
            mime="text/csv",
        )

    rows = [
        {"time": f.timestamp.strftime("%H:%M:%S"), "action": "EXTRACT",
         "content": f.content, "conf": f.confidence}
        for f in facts
    ]
    merged = {r["content"]: r for r in rows}
    for lr in ledger:
        merged.setdefault(lr["content"], lr)
    final = list(merged.values())
    if high_only:
        final = [r for r in final if r["conf"] >= 0.80]
    final.sort(key=lambda r: -r["conf"])

    st.markdown(_ledger_table(final[:12]), unsafe_allow_html=True)

    snapshot = [
        {
            "id": f.id[:8] + "…",
            "content": f.content[:56] + ("…" if len(f.content) > 56 else ""),
            "confidence": round(f.confidence, 3),
            "timestamp": f.timestamp.strftime("%Y-%m-%d %H:%M UTC"),
        }
        for f in facts[:30]
    ]
    with st.expander(f"原始 JSON · {len(snapshot)} 条", expanded=False):
        st.json(snapshot, expanded=False)


def _memory_csv(facts: List[MemoryFact]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "time_utc", "content", "confidence"])
    for f in facts:
        w.writerow([f.id, f.timestamp.isoformat(), f.content, f.confidence])
    return buf.getvalue()


def tab_console(memory: MemoryManager, executor) -> None:
    for msg in st.session_state["messages"]:
        role = msg["role"]
        with st.chat_message(role, avatar="🤖" if role == "assistant" else "👤"):
            st.markdown(
                f'<div class="chip {"blue" if role == "assistant" else "dim"}">'
                f'{"智能体" if role == "assistant" else "用户"}</div>{msg["content"]}',
                unsafe_allow_html=True,
            )

    if not st.session_state["messages"]:
        st.markdown(
            '<div style="color:var(--faint); font-size:.84rem; padding:18px 4px;">'
            '在页面底部的输入框下发任务。每轮输入会写入记忆台账并触发 '
            'AgentExecutor（规划 → ReAct → Critic 验证）。</div>',
            unsafe_allow_html=True,
        )

    # 故障注入开关：勾选后下一次任务的第一个子任务会被 Critic 拒绝，
    # 触发 自动重试 → 重规划 → 成功 的完整自纠错演示链路。
    if st.session_state.get("_demo_mode", True):
        st.checkbox(
            "演示自纠错（下一次任务：Critic 先拒绝，再自动重试成功）",
            key="demo_fault",
            value=False,
        )


def tab_retrieval(memory: MemoryManager) -> None:
    st.markdown(
        '<span class="chip dim">混合检索 · BM25 + 向量 → RRF (k=60)</span>',
        unsafe_allow_html=True,
    )
    query = st.text_input("query", placeholder="输入查询，体验混合检索排序，例如：pytest / 截止时间 / 冲突消解…",
                          label_visibility="collapsed")
    if query and isinstance(query, str) and query.strip():
        hits = memory.recall(query, top_k=6)
        if hits:
            rows = [
                {"time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                 "action": "RRF", "content": h.fact.content, "conf": h.score}
                for h in hits
            ]
            trs = []
            for i, (r, h) in enumerate(zip(rows, hits), 1):
                pct = int(round(min(1.0, h.score) * 100))
                trs.append(
                    f'<tr>'
                    f'<td class="t-time">#{i}</td>'
                    f'<td><span class="act fuse">RRF {h.score:.3f}</span></td>'
                    f'<td class="c-content" title="{h.fact.content}">{h.fact.content}</td>'
                    f'<td class="v-conf">{h.fact.confidence:.2f}'
                    f'<span class="bar"><i style="width:{pct}%"></i></span></td>'
                    f'<td><span class="tok mnt"><span class="c">◎</span>命中</span></td>'
                    f'</tr>'
                )
            st.markdown(
                '<div class="ledger-wrap"><table class="ledger">'
                '<thead><tr><th>排名</th><th>得分</th><th>内容</th><th>置信度</th><th>状态</th></tr></thead>'
                f'<tbody>{"".join(trs)}</tbody></table></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="chip rose">无命中</span>', unsafe_allow_html=True)


def tab_ab() -> None:
    ctl = st.columns([1.2, 1, 1, 2.6])
    with ctl[0]:
        n = st.slider("样本 / 组", min_value=10, max_value=80,
                      value=st.session_state["ab_n"], step=5)
    with ctl[1]:
        regenerate = st.button("⟳ 重新生成")
    with ctl[2]:
        real = st.button("▶ 真实基准测试", type="primary")

    if real:
        try:
            with st.spinner("run_benchmark() 执行中（需 LLM）…"):
                from app.evaluation.run_benchmark import run_benchmark  # noqa: PLC0415
                st.session_state["ab_results"] = run_benchmark(quick=True)
                st.success("真实基准测试完成")
        except Exception as exc:  # noqa: BLE001
            logger.warning("真实基准测试失败: %s", exc)
            st.warning(f"真实基准测试失败，已保留当前数据：{str(exc)[:80]}")

    if regenerate or st.session_state.get("ab_results") is None or n != st.session_state["ab_n"]:
        st.session_state["ab_n"] = n
        with st.spinner("生成数据 → Welch t 检验…"):
            st.session_state["ab_results"] = _generate_demo_ab_results(n=n)

    ab: Optional[ABTestReport] = st.session_state.get("ab_results")
    if not ab:
        st.info("暂无 A/B 结果")
        return

    dark_path = str(PROJECT_ROOT / "evaluation" / "results_dark.png")
    _render_dark_chart(ab, dark_path)

    body = st.columns([1.55, 1])
    with body[0]:
        st.image(dark_path, use_column_width=True)
    with body[1]:
        rows = [
            {
                "指标": c.label,
                "A": round(c.mean_a, 3),
                "B": round(c.mean_b, 3),
                "Δ(B−A)": f"{c.diff:+.3f}",
                "p": f"{c.p_value:.4f}" + (" *" if c.significant else ""),
            }
            for c in ab.comparisons
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        improve = ab.significant_improvements()
        degrade = ab.significant_degradations()
        st.markdown(
            f'<div style="margin-top:10px;">'
            f'<span class="chip">改善 ×{len(improve)}</span>'
            f'<span class="chip rose">退化 ×{len(degrade)}</span>'
            f'<span class="chip dim">α={ab.alpha} · n={ab.n_a}/{ab.n_b}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if ab.report_path and Path(ab.report_path).exists():
            with st.expander("Markdown 评估报告", expanded=False):
                st.markdown(Path(ab.report_path).read_text(encoding="utf-8"))


def render_footer() -> None:
    st.markdown(
        """
<div class="powered">
  <div class="pl">
    <div class="small">技术内核</div>
    <div class="big">统计优化</div>
  </div>
  <div class="pr">
    MEMSCAN 是面向长程记忆与自纠错智能体的可视化控制台：
    通过冲突消解、序贯检验（SPRT）与 SPC 过程监控，
    让智能体的每一步统计决策都可审计、可复现。
  </div>
</div>
<div class="wm">MEMSCAN</div>
<div class="foot-links">
  <div class="grp"><span>记忆</span><span>智能体</span><span>评估</span><span>部署</span></div>
  <div class="cp">memscan@2026 · Welch t 检验 · SPRT · SPC</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# 6. 入口
# ============================================================

def _handle_prompt(memory: MemoryManager, executor, prompt: str) -> None:
    """全局任务入口：写记忆 → 执行智能体 → 反馈 + 经验落盘（任何页签可用）。"""
    llm = st.session_state.get("_llm")

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f'<div class="chip dim">用户</div>{prompt}', unsafe_allow_html=True)

    action, conf = ingest_chat_turn(memory, prompt)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            f'<span class="chip">记忆库 → 已写入</span>'
            f'<span class="chip amber">置信度 {conf:.2f}</span>',
            unsafe_allow_html=True,
        )
        # 故障注入（仅演示模式）：第一个子任务被 Critic 拒绝 → 触发自动重试。
        # 注意：Streamlit 每次 rerun 都会重新定义本模块的类，session 中缓存的
        # llm 实例属于"旧类"，isinstance 判定恒为 False，因此这里用 duck typing。
        injecting = bool(st.session_state.get("demo_fault")) and hasattr(
            llm, "_pending_failures"
        )
        if injecting:
            llm._pending_failures = 1
        with st.spinner("状态机执行中 · 规划 → 执行 → 验证…"):
            result = run_agent(executor, prompt)
        if injecting:
            llm._pending_failures = 0
            # 勾选框本轮已实例化，直接改其 session key 会抛异常；
            # 置标志，交由下一轮 rerun 顶部（实例化前）安全复位。
            st.session_state["fault_consume"] = True

        st.session_state["runs"] = st.session_state.get("runs", 0) + 1
        max_retries = max((r["retries"] for r in result["results"]), default=0)
        if result["status"] == "success":
            st.session_state["ok_runs"] = st.session_state.get("ok_runs", 0) + 1
            _append_ledger("EXTRACT", f"已完成任务：{prompt[:40]}", conf)
            if max_retries:
                _append_ledger("RETRY", f"自纠错：Critic 拒绝后第 {max_retries} 次重试成功", conf)

        if result["status"] == "success":
            chips = f'<span class="chip">成功 · {len(result["results"])} 个子任务</span>'
            chips += (
                f'<span class="chip amber">自纠错 ×{max_retries} 次重试成功</span>'
                if max_retries
                else f'<span class="chip dim">{len(result["transitions"])} 次状态流转</span>'
            )
            st.markdown(chips, unsafe_allow_html=True)
            with st.expander("执行轨迹", expanded=False):
                for r in result["results"]:
                    st.markdown(
                        f"- **{r['subtask_id']}** {r['description'][:44]}\n"
                        f"  - 答案：_{(r['answer'] or '∅')[:110]}_\n"
                        f"  - Critic `passed={r['passed']}` · 重试 `{r['retries']}`"
                    )
                st.code(" → ".join(result["transitions"]), language=None)
        else:
            st.markdown(
                f'<span class="chip rose">失败</span>'
                f'<span class="chip dim">{(result.get("error") or "未知错误")[:90]}</span>',
                unsafe_allow_html=True,
            )

        summary = (
            f"任务完成：本轮输入已写入记忆库（置信度 {conf:.2f}），"
            f"执行结果 **{'成功' if result['status'] == 'success' else '失败'}**，"
            f"共 {len(result['results'])} 个子任务。"
        )
        if max_retries:
            summary += f" 其中自纠错机制在 Critic 拒绝后自动重试 {max_retries} 次并成功。"
        st.session_state["messages"].append({"role": "assistant", "content": summary})
    st.rerun()


def main() -> None:
    inject_visual_layer()
    _init_state()

    # 上轮任务消费了故障注入 → 在勾选框实例化之前复位其 session key（安全窗口）
    if st.session_state.pop("fault_consume", False):
        st.session_state["demo_fault"] = False

    demo = st.session_state.get("_demo_mode", True)
    render_nav(demo)
    render_hero(demo)
    render_trio(st.session_state["_memory"])

    # 药丸导航：st.radio 原生实现（st.tabs 在自定义 CSS 下点击事件不可靠）
    page = st.radio(
        "页面导航",
        ["记忆台账", "控制台", "检索", "基准测试"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if page == "记忆台账":
        tab_ledger(st.session_state["_memory"])
    elif page == "控制台":
        tab_console(st.session_state["_memory"], st.session_state["_executor"])
    elif page == "检索":
        tab_retrieval(st.session_state["_memory"])
    elif page == "基准测试":
        tab_ab()

    # 全局任务输入：固定于页面底部，任何页签都可直接下发任务
    prompt = st.chat_input("输入任务描述，例如：调研三个竞品并输出对比报告…")
    if prompt and isinstance(prompt, str) and prompt.strip():
        _handle_prompt(st.session_state["_memory"], st.session_state["_executor"], prompt)

    render_footer()


if __name__ == "__main__":
    main()
