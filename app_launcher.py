import sys
import time
import os
import signal
import threading
from streamlit.web import cli as stcli
from streamlit.runtime import get_instance


def watchdog():
    """ブラウザの接続状態を監視し、誰もいなくなったらプロセスを終了する"""
    # 1. サーバーが起動するまで待機
    runtime = None
    while runtime is None:
        time.sleep(1)
        try:
            runtime = get_instance()
        except RuntimeError:
            # Streamlit raises until the server runtime exists; it does not return None.
            pass

    # 2. 最初のブラウザが接続されるまで待機（猶予期間）
    # 旧: runtime._session_info_by_id → 現行 Streamlit は runtime._session_mgr で管理
    session_mgr = runtime._session_mgr
    print("⏳ ブラウザの起動を待機中...")
    while session_mgr.num_active_sessions() == 0:
        time.sleep(1)

    print("🚀 ブラウザが接続されました。監視を開始します。")

    # 3. 接続がゼロになったら終了
    while True:
        time.sleep(2)  # 2秒おきにチェック
        session_count = session_mgr.num_active_sessions()
        if session_count == 0:
            print("📴 全てのブラウザが閉じられました。サーバーを終了します...")
            os.kill(os.getpid(), signal.SIGTERM)
            break


if __name__ == "__main__":
    monitor_thread = threading.Thread(target=watchdog, daemon=True)
    monitor_thread.start()

    sys.argv = ["streamlit", "run", "main_st.py", "--server.headless", "false"]
    sys.exit(stcli.main())
