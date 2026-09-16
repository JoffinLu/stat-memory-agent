# -*- coding: utf-8 -*-
"""回归测试：四个页签可点、底部输入全局可用、故障注入自纠错闭环。"""
import sys
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8765/"
SHOT = r"D:\Desktop\Data\项目类\2026年9月15日——AI研究员（智能体方向）\stat_memory_agent\screenshots"
PAGES = ["记忆台账", "控制台", "检索", "基准测试"]


def main() -> None:
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_selector("text=记忆台账", timeout=30000)
        page.wait_for_timeout(2000)

        # 1) Hero 文案已改（不再假装是搜索框）
        hero = page.locator(".hero-search .ph")
        if hero.count():
            txt = hero.first.inner_text()
            results.append(("hero 文案指向底部输入框", "底部" in txt))
            print("hero text:", txt)

        # 2) 四个页签依次点击
        for name in PAGES:
            page.locator("label", has_text=name).first.click()
            page.wait_for_timeout(1500)
            body = page.inner_text("body")
            results.append((f"页签[{name}]可点击", len(body) > 500))
            print(f"page {name}: body_len={len(body)}")

        # 3) 默认页签（记忆台账）也应能看到底部输入框 —— 本次修复核心
        page.locator("label", has_text="记忆台账").first.click()
        page.wait_for_timeout(1200)
        ta = page.locator("div[data-testid='stChatInput'] textarea")
        results.append(("记忆台账页可见全局输入框", ta.count() > 0 and ta.first.is_visible()))
        print("chat_input on ledger page:", ta.count(), ta.first.is_visible() if ta.count() else "-")

        # 4) 故障注入闭环：勾选自纠错 → 发任务 → 验证"重试成功"
        page.locator("label", has_text="控制台").first.click()
        page.wait_for_timeout(2500)
        cb = page.locator("label", has_text="演示自纠错")
        print("checkbox count:", cb.count())
        clicked = False
        for _ in range(3):  # 最多重试 3 次勾选
            if not cb.count():
                page.wait_for_timeout(1500)
                continue
            cb.first.click()
            page.wait_for_timeout(1500)
            checked = page.evaluate(
                "() => { const els = document.querySelectorAll("
                "'div[data-testid=\"stCheckbox\"] input[type=checkbox]');"
                " return els.length ? els[els.length-1].checked : null; }"
            )
            print("checkbox checked:", checked)
            clicked = bool(checked)
            if clicked:
                break
        results.append(("故障注入勾选成功", clicked))

        if ta.count():
            ta.first.click()
            ta.first.fill("调研三个竞品并输出对比报告")
            ta.first.press("Enter")
            page.wait_for_timeout(8000)  # 注入故障多一轮 Critic + 重试
            body = page.inner_text("body")
            marker = ("次重试成功" in body) or ("自纠错机制在 Critic 拒绝后" in body)
            if not marker:
                i = body.find("任务完成")
                print("diagnostic:", body[i:i + 120] if i >= 0 else body[-150:])
            page.screenshot(path=SHOT + r"\pw_fault_inject.png")
            no_crash = ("StreamlitAPIException" not in body) and ("Uncaught" not in body)
            results.append(("自纠错演示链路触发", marker))
            results.append(("无 avatar 崩溃异常", no_crash))
            print("fault-inject markers:", marker, "| no crash:", no_crash)
            page.wait_for_timeout(1500)
        else:
            results.append(("自纠错演示链路触发", False))

        # 5) 回台账页看 RETRY 芯片
        page.locator("label", has_text="记忆台账").first.click()
        page.wait_for_timeout(1500)
        body = page.inner_text("body")
        results.append(("台账出现 RETRY/重试记录", "重试" in body))
        page.screenshot(path=SHOT + r"\pw_ledger_retry.png")
        browser.close()

    print("\n===== 回归结果 =====")
    all_ok = True
    for name, ok in results:
        print(("PASS " if ok else "FAIL ") + name)
        all_ok = all_ok and ok
    print("ALL:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
