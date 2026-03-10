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
        runtime = get_instance()

    # 2. 最初のブラウザが接続されるまで待機（猶予期間）
    print("⏳ ブラウザの起動を待機中...")
    while not runtime._session_info_by_id:
        time.sleep(1)
    
    print("🚀 ブラウザが接続されました。監視を開始します。")

    # 3. 接続がゼロになったら終了
    while True:
        time.sleep(2) # 2秒おきにチェック
        session_count = len(runtime._session_info_by_id)
        if session_count == 0:
            print("📴 全てのブラウザが閉じられました。サーバーを終了します...")
            # 自分自身のプロセスに終了信号を送る
            os.kill(os.getpid(), signal.SIGTERM)
            break

if __name__ == "__main__":
    # 監視スレッドをデーモンとして開始
    monitor_thread = threading.Thread(target=watchdog, daemon=True)
    monitor_thread.start()

    # Streamlitをプログラム内から実行
    # 引数は 'streamlit run main_st.py' と同等
    sys.argv = ["streamlit", "run", "main_st.py", "--server.headless", "false"]
    sys.exit(stcli.main())