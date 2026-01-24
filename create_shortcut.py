"""
デスクトップショートカットを作成するスクリプト
失敗してもアプリは使えるので、エラーハンドリングは緩め
"""
import os
from pathlib import Path

def create_shortcut_simple():
    """シンプルな方法でショートカット作成（外部ライブラリ不要）"""
    try:
        # デスクトップパスを取得
        desktop = Path.home() / "Desktop"
        if not desktop.exists():
            desktop = Path.home() / "デスクトップ"
        
        # 現在のディレクトリ
        current_dir = Path(__file__).parent.absolute()
        run_bat = current_dir / "run.bat"
        icon_file = current_dir / "icon.ico"
        
        # ショートカット（.lnk）を作成するVBScript
        vbs_script = current_dir / "create_shortcut.vbs"
        
        vbs_content = f'''Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{desktop}\\画像ROIツール.lnk"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{run_bat}"
oLink.WorkingDirectory = "{current_dir}"
oLink.Description = "画像ROI選択ツール"
'''
        
        if icon_file.exists():
            vbs_content += f'oLink.IconLocation = "{icon_file}"\n'
        
        vbs_content += 'oLink.Save\n'
        
        # VBScriptを保存
        with open(vbs_script, 'w', encoding='utf-8') as f:
            f.write(vbs_content)
        
        # VBScriptを実行
        os.system(f'cscript //nologo "{vbs_script}"')
        
        # 一時ファイルを削除
        try:
            vbs_script.unlink()
        except:
            pass
        
        print("✅ ショートカットを作成しました")
        return True
        
    except Exception as e:
        print(f"⚠️  ショートカット作成エラー: {e}")
        print("   run.bat を直接実行してください")
        return False

if __name__ == "__main__":
    create_shortcut_simple()