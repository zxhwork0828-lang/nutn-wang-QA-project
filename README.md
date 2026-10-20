# Streamlit 手動上傳版本

此資料夾的 app.py 已整合問答、分類面板與語意搜尋，部署不需要 src 資料夾。

1. 解壓縮後，將此資料夾內的檔案與 .streamlit/config.toml 上傳至 GitHub 專案根目錄。
2. 使用這裡的 app.py 與 main.py 覆蓋 GitHub 上的同名舊檔。
3. Streamlit Cloud 入口建議選 app.py；既有入口為 main.py 也可繼續使用。
4. 在 Streamlit Secrets 設定下面的資料庫連線，重新啟動應用程式。

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-server-side-secret"
QA_MIN_SIMILARITY = "0.70"
```

SUPABASE_URL 是專案根網址，不含 /rest/v1/。金鑰只填在 Streamlit Secrets。
資料庫需已執行原專案的 qa.sql 與 supabase_search.sql，具備 public.qa 與 match_qa RPC。
QA.csv 供左側分類面板使用；未設定 Supabase 時可使用隨附 qa.sql 的本機向量。

雲端 Python 建議使用 3.12，依賴使用 requirements.txt。
模型首次使用會自動下載，不要設定指向本機 Windows 路徑的 QA_MODEL_DIR。
本機可執行 python -m streamlit run app.py。

此部署檔由 scripts/build_streamlit_bundle.py 從原始碼產生；修改原始碼後請重新執行產生器。
