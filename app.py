"""Streamlit entry point: uv run streamlit run app.py."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Render the app, with an actionable message for incomplete manual uploads."""
    required = [
        ROOT / "src" / "wang_project" / name
        for name in ("__init__.py", "ui.py", "qa.py", "catalog.py")
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        import streamlit as st

        st.error("部署檔案不完整：請將完整的 src/wang_project 資料夾上傳至 GitHub，並保留目錄結構與檔名大小寫。")
        st.write("缺少以下檔案：")
        st.code("\n".join(missing), language="text")
        return

    # Community Cloud runs the script without installing this src-layout project.
    sys.path.insert(0, str(ROOT / "src"))
    from wang_project.ui import render_app

    render_app()


if __name__ == "__main__":
    main()
