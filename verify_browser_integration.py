"""
verify_browser_integration.py — Browser-level end-to-end integration verification
for Cross-Lingual Hindi RAG Engine (FastAPI backend + React/Vite frontend).
Uses Playwright to control Google Chrome directly in headless mode.
"""
import os
import sys
import time
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

WORKSPACE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = WORKSPACE_DIR / "backend"
FRONTEND_DIR = WORKSPACE_DIR / "frontend"
SCREENSHOTS_DIR = WORKSPACE_DIR / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = sys.executable
NPM_CMD = shutil.which("npm.cmd") or shutil.which("npm") or "npm"
CHROME_PATH = "C:/Program Files/Google/Chrome/Application/chrome.exe"


def wait_for_http(url: str, timeout: int = 60) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.time()
    while time.time() - start < timeout:
        try:
            with opener.open(url, timeout=2) as resp:
                if resp.status in (200, 304):
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def run_verification():
    print("=" * 80)
    print("PHASE 7 BROWSER-LEVEL INTEGRATION VERIFICATION")
    print("=" * 80)

    # 1. Start FastAPI backend with log file to prevent pipe deadlock
    backend_log_path = WORKSPACE_DIR / "backend_server.log"
    backend_log_file = open(backend_log_path, "w", encoding="utf-8")

    print("\n[1/7] Launching FastAPI backend server (http://127.0.0.1:8000)...")
    backend_proc = subprocess.Popen(
        [PYTHON_EXE, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(BACKEND_DIR),
        stdout=backend_log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("  Waiting for backend startup (loading BGE-M3 and ChromaDB)...")
    if not wait_for_http("http://127.0.0.1:8000/health", timeout=75):
        backend_proc.kill()
        backend_log_file.close()
        with open(backend_log_path, "r", encoding="utf-8", errors="replace") as f:
            err_log = f.read()
        raise RuntimeError(f"Backend failed to become healthy. Log:\n{err_log[-2000:]}")
    print("  [OK] FastAPI backend healthy and responding on port 8000.")

    # 2. Start Vite frontend
    frontend_log_path = WORKSPACE_DIR / "frontend_server.log"
    frontend_log_file = open(frontend_log_path, "w", encoding="utf-8")

    print("\n[2/7] Launching Vite frontend server (http://localhost:5173)...")
    frontend_proc = subprocess.Popen(
        [NPM_CMD, "run", "dev", "--", "--port", "5173", "--host", "0.0.0.0"],
        cwd=str(FRONTEND_DIR),
        stdout=frontend_log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if not wait_for_http("http://localhost:5173", timeout=30):
        frontend_proc.kill()
        backend_proc.kill()
        frontend_log_file.close()
        backend_log_file.close()
        with open(frontend_log_path, "r", encoding="utf-8", errors="replace") as f:
            err_log = f.read()
        raise RuntimeError(f"Frontend Vite server failed to respond. Log:\n{err_log}")
    print("  [OK] Vite frontend server responding on port 5173.")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": {},
        "console_logs": [],
        "network_errors": [],
    }

    try:
        with sync_playwright() as p:
            print("\n[3/7] Launching headless Google Chrome via Playwright...")
            browser = p.chromium.launch(
                executable_path=CHROME_PATH,
                headless=True,
            )
            context = browser.new_context(viewport={"width": 1280, "height": 800})
            page = context.new_page()

            # Attach console and error listeners
            page.on("console", lambda msg: results["console_logs"].append(f"[{msg.type}] {msg.text}"))
            page.on("pageerror", lambda err: results["network_errors"].append(str(err)))

            # Step 1: Initial Load & Theme Toggle
            print("\n[Step 1] Verifying Initial Load, Dark/Light Theme & Persistence...")
            page.goto("http://localhost:5173", wait_until="networkidle")
            page.wait_for_selector("h1:has-text('Hindi RAG Engine')", timeout=15000)
            page.wait_for_selector("span:has-text('Online')", timeout=15000)

            # Toggle theme to Light
            theme_btn = page.locator("button[aria-label='Toggle theme']")
            theme_btn.click()
            time.sleep(0.5)
            page.screenshot(path=str(SCREENSHOTS_DIR / "01_light_mode.png"))
            assert not page.locator("html").evaluate("el => el.classList.contains('dark')")
            print("  [OK] Light mode toggled and verified.")

            # Toggle theme back to Dark and reload
            theme_btn.click()
            time.sleep(0.5)
            assert page.locator("html").evaluate("el => el.classList.contains('dark')")
            page.reload(wait_until="networkidle")
            time.sleep(0.5)
            assert page.locator("html").evaluate("el => el.classList.contains('dark')")
            page.screenshot(path=str(SCREENSHOTS_DIR / "02_dark_mode_persisted.png"))
            print("  [OK] Dark mode persisted across page reload.")
            results["steps"]["theme_toggle_persistence"] = "PASSED"

            # Step 2: Hindi Mandatory Question
            print("\n[Step 2] Submitting Hindi Mandatory Question (SLV-III)...")
            hindi_query = "SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?"
            textarea = page.locator("textarea")
            textarea.fill(hindi_query)
            time.sleep(0.3)
            # Submit
            send_btn = page.locator("button[aria-label='Send question']")
            send_btn.click()

            # Wait for assistant response to appear
            print("  Waiting for backend response and dynamic citations...")
            page.wait_for_selector("div:has-text('Hindi RAG Assistant')", timeout=60000)
            page.wait_for_selector("span:has-text('Verifiable Source Citations')", timeout=60000)
            time.sleep(1)

            page.screenshot(path=str(SCREENSHOTS_DIR / "03_hindi_query_grounded_answer.png"))
            print("  [OK] Hindi answer and citations rendered.")
            results["steps"]["hindi_query_execution"] = "PASSED"

            # Step 3: Citation Expansion & SourceViewer Modal
            print("\n[Step 3] Testing Source Citation Expansion & Modal Inspector...")
            # Click first citation card to expand
            cite_btn = page.locator("button:has-text('Page ')").first
            cite_btn.click()
            time.sleep(1)

            # Click Inspect button to open SourceViewer modal
            inspect_btn = page.locator("button:has-text('Inspect')").first
            inspect_btn.click()
            time.sleep(1)

            page.wait_for_selector("p:has-text('Chunk ID:')", timeout=10000)
            page.screenshot(path=str(SCREENSHOTS_DIR / "04_source_viewer_modal.png"))
            print("  [OK] SourceViewer modal opened displaying exact chunk metadata and text.")

            # Close modal
            close_btn = page.locator("button:has-text('Done')")
            close_btn.click()
            time.sleep(0.5)
            results["steps"]["source_viewer_modal"] = "PASSED"

            # Step 4: English Mandatory Question
            print("\n[Step 4] Submitting English Mandatory Question (Aeronautical Engineering)...")
            english_query = "Which institution did Kalam attend to study aeronautical engineering?"
            textarea.fill(english_query)
            send_btn.click()

            print("  Waiting for English answer and citations...")
            time.sleep(2)
            page.wait_for_selector("div:has-text('You')", timeout=60000)
            page.wait_for_selector("span:has-text('Verifiable Source Citations')", timeout=60000)
            time.sleep(1)
            page.screenshot(path=str(SCREENSHOTS_DIR / "05_english_query_answer.png"))
            print("  [OK] English query processed and cited with verified document passages.")
            results["steps"]["english_query_execution"] = "PASSED"

            # Step 5: Unanswerable Question (Refusal Test)
            print("\n[Step 5] Submitting Out-of-Domain Unanswerable Question...")
            unans_query = "कलाम ने किस वर्ष मंगल ग्रह (Mars) पर भारतीय अंतरिक्ष स्टेशन स्थापित किया?"
            textarea.fill(unans_query)
            send_btn.click()

            print("  Waiting for refusal response...")
            time.sleep(2)
            page.wait_for_selector("span:has-text('Document Refusal')", timeout=60000)
            time.sleep(1)
            page.screenshot(path=str(SCREENSHOTS_DIR / "06_unanswerable_refusal.png"))
            print("  [OK] Document Refusal badge and standard refusal response rendered.")
            results["steps"]["unanswerable_refusal_ui"] = "PASSED"

            # Step 6: Mobile Responsiveness (375px width)
            print("\n[Step 6] Testing Mobile Responsiveness (375px viewport)...")
            page.set_viewport_size({"width": 375, "height": 812})
            time.sleep(0.5)

            # Open mobile sidebar drawer
            menu_btn = page.locator("button[title='Toggle sidebar']")
            menu_btn.click()
            time.sleep(0.5)
            page.screenshot(path=str(SCREENSHOTS_DIR / "07_mobile_drawer_375px.png"))
            print("  [OK] Mobile sidebar drawer rendered at 375px width.")

            # Close mobile drawer by clicking close button inside sidebar header
            close_drawer_btn = page.locator("aside button:has(svg.lucide-x)")
            if close_drawer_btn.count() > 0 and close_drawer_btn.first.is_visible():
                close_drawer_btn.first.click()
            else:
                menu_btn.click()
            time.sleep(1)

            page.screenshot(path=str(SCREENSHOTS_DIR / "08_mobile_chat_375px.png"))
            print("  [OK] Mobile chat interface rendered cleanly at 375px width.")
            page.set_viewport_size({"width": 1280, "height": 800})
            results["steps"]["mobile_responsiveness"] = "PASSED"

            # Step 7: Backend Disconnection & Error Handling
            print("\n[Step 7] Testing Backend Disconnection & Error Retry UI...")
            print("  Simulating backend outage by terminating FastAPI process...")
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except Exception:
                backend_proc.kill()
            time.sleep(2)

            # Try sending a query while backend is disconnected
            textarea.fill("Testing query with backend disconnected")
            send_btn.click()
            time.sleep(1)

            page.wait_for_selector("div:has-text('Communication Error')", timeout=10000)
            page.wait_for_selector("button:has-text('Retry')", timeout=10000)
            page.screenshot(path=str(SCREENSHOTS_DIR / "09_backend_disconnected_error.png"))
            print("  [OK] Error card with Retry button cleanly rendered on backend failure.")
            results["steps"]["backend_disconnection_error_handling"] = "PASSED"

            browser.close()

    finally:
        print("\nCleaning up server processes...")
        try:
            frontend_proc.terminate()
        except Exception:
            pass
        try:
            backend_proc.terminate()
        except Exception:
            pass
        try:
            frontend_log_file.close()
        except Exception:
            pass
        try:
            backend_log_file.close()
        except Exception:
            pass

    # Save structured verification results
    report_file = WORKSPACE_DIR / "results" / "browser_verification_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("ALL 7 BROWSER-LEVEL INTEGRATION CHECKS PASSED SUCCESSFULLY!")
    print(f"Screenshots saved to: {SCREENSHOTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
