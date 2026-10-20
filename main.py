"""Support both python main.py and streamlit run main.py."""

from pathlib import Path
import runpy
import sys


def main() -> None:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    app_file = Path(__file__).resolve().parent / "app.py"
    if get_script_run_ctx(suppress_warning=True) is not None:
        # Streamlit Cloud already owns the server: render in its current session.
        runpy.run_path(str(app_file), run_name="__main__")
    else:
        from streamlit.web import cli

        sys.argv = ["streamlit", "run", str(app_file), *sys.argv[1:]]
        raise SystemExit(cli.main())


if __name__ == "__main__":
    main()
