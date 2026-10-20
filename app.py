"""Generated Streamlit deployment app; rebuild with scripts/build_streamlit_bundle.py."""
from __future__ import annotations
import streamlit as st
import csv
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
import httpx
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from pglast import ast, parse_sql
from tokenizers import Tokenizer
ROOT = Path(__file__).resolve().parent
MODEL_ID = 'Xenova/bge-small-zh-v1.5'
MODEL_REVISION = '75c43b069aac4d136ba6bc1122f995fedcfd2781'
MODEL_SHA256 = 'b9837c19ce154ff0726d398ee77abbc03a7faf0476c6f93016c84e531be7ebb5'
DIMENSIONS = 512
logger = logging.getLogger(__name__)
TEST_NAMES = {'CBC': '全血球計數', 'RBC': '紅血球計數', 'WBC': '白血球計數', 'Hb': '血紅素', 'Hct': '血比容', 'MCV': '平均紅血球體積', 'MCH': '平均紅血球血紅素量', 'MCHC': '平均紅血球血紅素濃度', 'RDW': '紅血球分布寬度', 'PLT': '血小板', 'HbA1c': '糖化血色素', 'BUN': '血中尿素氮', 'eGFR': '估算腎絲球過濾率', 'ALT': '丙胺酸轉胺酶', 'AST': '天門冬胺酸轉胺酶', 'ALP': '鹼性磷酸酶', 'GGT': '麩胺酸轉移酶', 'LDL': '低密度脂蛋白膽固醇', 'HDL': '高密度脂蛋白膽固醇', 'TSH': '促甲狀腺激素', 'CRP': 'C 反應蛋白', 'ESR': '紅血球沉降速率', 'PT': '凝血酶原時間', 'INR': '國際標準化比值', 'aPTT': '活化部分凝血活酶時間', 'TIBC': '總鐵結合能力', 'CO2': '總二氧化碳', 'Creatinine': '肌酸酐', 'Albumin': '白蛋白', 'Ferritin': '鐵蛋白'}
_test_names_lower = {key.lower(): value for (key, value) in TEST_NAMES.items()}
_test_pattern = re.compile('(?<![a-zA-Z0-9])(' + '|'.join(sorted(TEST_NAMES, key=len, reverse=True)) + ')(?![a-zA-Z0-9])', re.IGNORECASE)

def prepare_query(text: str) -> str:
    """Expand lab abbreviations the fixed model may otherwise tokenize as unknown."""

    def expand(match: re.Match) -> str:
        name = _test_names_lower[match.group(0).lower()]
        return match.group(0) if name in text else f'{name} {match.group(0)}'
    return _test_pattern.sub(expand, text)

class QAError(Exception):
    """An actionable error safe to display to the user."""

@dataclass(frozen=True)
class Match:
    question: str
    answer: str
    source_url: str
    similarity: float

class Encoder:
    """Pinned INT8 BERT model, CLS pooling, and L2 normalization."""

    def __init__(self) -> None:
        model_dir = os.getenv('QA_MODEL_DIR')
        if model_dir:
            folder = Path(model_dir).expanduser().resolve()
            model_path = folder / 'model_int8.onnx'
            tokenizer_path = folder / 'tokenizer.json'
        else:
            model_path = Path(hf_hub_download(MODEL_ID, 'onnx/model_int8.onnx', revision=MODEL_REVISION))
            tokenizer_path = Path(hf_hub_download(MODEL_ID, 'tokenizer.json', revision=MODEL_REVISION))
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise QAError('模型檔案版本不符，請重新下載指定版本。')
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])

    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.encode(text)
        if len(tokens.ids) > 512:
            raise QAError('問題太長，請縮短內容並一次詢問一個檢驗項目。')
        inputs = {'input_ids': np.asarray([tokens.ids], dtype=np.int64), 'attention_mask': np.asarray([tokens.attention_mask], dtype=np.int64), 'token_type_ids': np.asarray([tokens.type_ids], dtype=np.int64)}
        hidden = self.session.run(None, {item.name: inputs[item.name] for item in self.session.get_inputs()})[0]
        vector = np.asarray(hidden[0, 0, :], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all() or norm == 0:
            raise QAError('目前無法處理這個問題，請稍後再試。')
        return vector / norm

class LocalIndex:
    """Read data literals from SQL without executing any SQL statements."""

    def __init__(self, folder: Path=ROOT) -> None:
        source = folder / 'QA.csv'
        sql = (folder / 'qa.sql').read_text(encoding='utf-8-sig')
        required_headers = {'Model': MODEL_ID, 'Model revision': MODEL_REVISION, 'Model file SHA256': MODEL_SHA256, 'Source SHA256': hashlib.sha256(source.read_bytes()).hexdigest()}
        for (name, expected) in required_headers.items():
            actual = re.search(f'^-- {re.escape(name)}: (.+)$', sql, re.MULTILINE)
            if not actual or actual.group(1).strip() != expected:
                raise QAError('QA.csv、qa.sql 或模型版本不一致，請重新產生對應的向量資料。')
        with source.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        inserts = [statement.stmt for statement in parse_sql(sql) if isinstance(statement.stmt, ast.InsertStmt) and statement.stmt.relation.schemaname == 'public' and (statement.stmt.relation.relname == 'qa')]
        values = [value for insert in inserts for value in insert.selectStmt.valuesLists]
        if not rows or len(rows) != len(values):
            raise QAError('問答筆數與向量筆數不一致。')
        vectors = []
        for (row, value) in zip(rows, values):
            if (row['question'], row['answer']) != (value[0].val.sval, value[1].val.sval):
                raise QAError('問答內容與向量資料不一致。')
            vectors.append(json.loads(value[3].arg.val.sval))
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(rows), DIMENSIONS) or not np.isfinite(matrix).all():
            raise QAError('向量維度或數值不正確。')
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise QAError('向量資料包含無效的零向量。')
        self.matrix = matrix / norms
        self.rows = rows

    def search(self, vector: np.ndarray, threshold: float) -> list[Match]:
        scores = self.matrix @ vector
        indices = np.argsort(-scores, kind='stable')[:3]
        return [Match(self.rows[i]['question'], self.rows[i]['answer'], self.rows[i].get('source_url', ''), float(scores[i])) for i in indices if scores[i] >= threshold]

class SupabaseIndex:
    """Call a server-side pgvector RPC; the secret never reaches the browser."""

    def __init__(self, url: str, key: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.username:
            raise QAError('SUPABASE_URL 必須是有效的 HTTPS 專案網址。')
        headers = {'apikey': key}
        if not key.startswith('sb_secret_'):
            headers['Authorization'] = f'Bearer {key}'
        self.client = httpx.Client(base_url=url.rstrip('/') + '/', headers=headers, timeout=15)

    def search(self, vector: np.ndarray, threshold: float) -> list[Match]:
        try:
            response = self.client.post('rest/v1/rpc/match_qa', json={'query_embedding': vector.tolist(), 'match_threshold': threshold, 'match_count': 3})
            response.raise_for_status()
            return [Match(str(row['question']), str(row['answer']), row.get('source_url') or '', float(row['similarity'])) for row in response.json()]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            logger.warning('Supabase search failed (%s)', type(error).__name__)
            raise QAError('暫時無法查詢資料庫，請稍後再試；管理者請確認連線與 match_qa 設定。') from None

class QAService:

    def __init__(self) -> None:
        self.threshold = float(os.getenv('QA_MIN_SIMILARITY', '0.70'))
        if not -1 <= self.threshold <= 1:
            raise QAError('QA_MIN_SIMILARITY 必須介於 -1 與 1。')
        url = os.getenv('SUPABASE_URL', '').strip()
        key = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '').strip()
        if bool(url) != bool(key):
            raise QAError('請同時設定 SUPABASE_URL 與 SUPABASE_SERVICE_ROLE_KEY。')
        self.index = SupabaseIndex(url, key) if url else LocalIndex()
        self.encoder = Encoder()
        logger.info('QA service ready (%s)', 'Supabase' if url else 'local')

    def answer(self, question: str) -> str:
        question = question.strip()
        if not question:
            return '請輸入想了解的血液檢驗項目或問題。'
        if len(question) > 1000:
            return '問題太長，請縮短內容並一次詢問一個檢驗項目。'
        matches = self.index.search(self.encoder.encode(prepare_query(question)), self.threshold)
        if not matches:
            return '目前的資料集中找不到足夠相關的回答。請寫出檢驗項目名稱與想了解的內容，例如「肌酸酐偏高代表什麼？」。'
        best = matches[0]
        answer = f'{best.answer}\n\n---\n參考問答：{best.question}'
        parsed = urlparse(best.source_url)
        if parsed.scheme == 'https' and parsed.netloc:
            answer += f'\n\n[查看參考來源](<{best.source_url}>)'
        else:
            answer += '\n\n此筆資料未附參考來源。'
        return answer
_service_lock = Lock()

@st.cache_resource(show_spinner=False)
def _create_service() -> QAService:
    return QAService()

def get_service() -> QAService:
    with _service_lock:
        return _create_service()
import csv
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass(frozen=True)
class Category:
    key: str
    title: str
    indicators: str
    aliases: str = ''
CATEGORIES = (Category('blood', '血液與貧血', 'CBC、紅白血球、血小板、血紅素、鐵蛋白', '造血 血球 骨髓 血色素 缺鐵'), Category('liver', '肝臟與膽道', 'ALT、AST、ALP、GGT、膽紅素、白蛋白、總蛋白', '肝功能 肝膽'), Category('kidney', '腎臟與尿酸', '肌酸酐、BUN、eGFR、尿酸', '腎功能 尿素氮 腎絲球 痛風'), Category('heart', '心臟、血管與血脂', '總膽固醇、LDL、HDL、三酸甘油脂、Troponin', '心血管 心肌 動脈'), Category('glucose', '血糖與胰臟代謝', '空腹血糖、HbA1c', '糖尿病 糖化血色素 胰島素'), Category('electrolytes', '電解質與酸鹼平衡', '鈉、鉀、氯、鎂、CO2、乳酸', '腎臟 體液 碳酸氫鹽'), Category('thyroid', '甲狀腺與內分泌', 'TSH、Free T3、Free T4', '激素 荷爾蒙'), Category('nutrition', '骨骼、鈣與營養', '血鈣、維生素 D、維生素 B12', '骨質 維他命 營養素'), Category('inflammation', '發炎與免疫', 'CRP、ESR', '感染 免疫系統'), Category('coagulation', '凝血功能', 'PT、INR、aPTT', '止血 出血'), Category('report', '報告判讀與抽血須知', '參考區間、H/L、單位、空腹與複檢', '檢驗報告 一般問題 採檢'))

@dataclass(frozen=True)
class CatalogQuestion:
    row_id: int
    question: str
    category: str
_RULES = tuple(((category, re.compile(pattern, re.IGNORECASE)) for (category, pattern) in (('glucose', '血糖|糖化血|HbA1c'), ('thyroid', '甲狀腺|TSH|Free T[34]'), ('inflammation', 'CRP|ESR|反應蛋白|沉降速率'), ('coagulation', '凝血|aPTT|PT/INR|(?<![a-z])INR(?![a-z])'), ('nutrition', '維生素|維他命|血鈣|血清鈣'), ('heart', '膽固醇|三酸甘油|脂蛋白|Troponin|肌鈣蛋白'), ('kidney', '肌酸酐|尿素氮|BUN|eGFR|尿酸|絲球.*過濾'), ('liver', '轉胺酶|轉氨酶|磷酸酶|膽紅素|白蛋白|總蛋白|ALT|AST|ALP|GGT'), ('electrolytes', '血鈉|血鉀|血氯|鉀離子|氯離子|鎂|CO2|二氧化碳|碳酸氫|乳酸'), ('blood', '血球|血小板|血紅素|血比容|血色素|嗜中性球|淋巴球|嗜酸性球|血清鐵|鐵蛋白|總鐵|Ferritin|TIBC|CBC|RBC|WBC|MCV|MCH|RDW|Hematocrit'))))

def classify_question(question: str) -> str:
    for (category, pattern) in _RULES:
        if pattern.search(question):
            return category
    return 'report'

def load_catalog(path: Path) -> tuple[CatalogQuestion, ...]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return tuple((CatalogQuestion(i, row['question'], classify_question(row['question'])) for (i, row) in enumerate(csv.DictReader(stream)) if row.get('question', '').strip()))

def filter_catalog(questions: tuple[CatalogQuestion, ...], search: str='') -> list[tuple[Category, tuple[CatalogQuestion, ...]]]:
    """Keep report order and support searching organ names, markers or questions."""
    terms = search.casefold().split()
    groups = []
    for category in CATEGORIES:
        category_text = f'{category.key} {category.title} {category.aliases}'.casefold()
        matching = tuple((item for item in questions if item.category == category.key and all((term in f'{category_text} {item.question.casefold()}' for term in terms))))
        if matching:
            groups.append((category, matching))
    return groups
import logging
import os
import streamlit as st
logger = logging.getLogger(__name__)
EXAMPLES = ['糖化血色素 HbA1c 反映什麼？', '肌酸酐偏高一定代表腎衰竭嗎？', '血小板偏低代表什麼？', 'ALT 偏高可能有哪些原因？']

def configure_secrets() -> None:
    """Read optional Streamlit Secrets; explicit environment variables take priority."""
    try:
        for name in ('SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'QA_MIN_SIMILARITY', 'QA_MODEL_DIR'):
            if name not in os.environ and name in st.secrets:
                os.environ[name] = str(st.secrets[name])
    except FileNotFoundError:
        pass

def respond(message: str) -> str:
    """Answer one complete blood-test question using the shared, cached QA service."""
    if not message.strip():
        return '請輸入想了解的血液檢驗項目或問題。'
    try:
        return get_service().answer(message)
    except QAError as error:
        return str(error)
    except Exception as error:
        logger.warning('QA request failed (%s)', type(error).__name__)
        return '服務暫時無法回覆，請稍後再試。'

def choose_example(question: str) -> None:
    st.session_state['pending_question'] = question

def clear_chat() -> None:
    st.session_state['messages'] = []
    st.session_state.pop('pending_question', None)

def render_question_panel() -> None:
    st.subheader('問題分類')
    st.caption('依常見健檢報告順序排列；各院所可能不同。點選分類展開，再點選問題即可查詢。')
    search = st.text_input('搜尋器官、指數或問題', placeholder='例如：腎臟、eGFR、膽固醇', key='catalog_search')
    try:
        questions = load_catalog(ROOT / 'QA.csv')
    except (OSError, ValueError, KeyError):
        st.info('暫時無法載入問題清單，仍可在右側輸入問題。')
        return
    groups = filter_catalog(questions, search)
    count = sum((len(items) for (_, items) in groups))
    st.caption(f'{count} / {len(questions)} 題')
    if not groups:
        st.info('找不到符合的問題，請換個檢驗名稱或縮寫。')
    for (category, items) in groups:
        number = next((i for (i, item) in enumerate(CATEGORIES, 1) if item.key == category.key))
        with st.expander(f'{number:02d} · {category.title}（{len(items)}）', expanded=bool(search.strip())):
            st.caption(category.indicators)
            for item in items:
                st.button(item.question, key=f'catalog_question_{item.row_id}', on_click=choose_example, args=(item.question,), width='stretch')
    st.caption('部分指標涉及多種器官，此分類供查找問題使用。')

def render_app() -> None:
    st.set_page_config(page_title='血液檢驗 QA', page_icon='🩸', layout='centered', initial_sidebar_state='expanded')
    configure_secrets()
    st.session_state.setdefault('messages', [])
    st.title('🩸 血液檢驗 QA')
    st.write('了解檢驗項目與異常數值的常見意義。輸入完整問題，即可查詢現有問答資料。')
    st.caption('每次請完整描述檢驗項目；個人結果仍須結合參考區間、症狀與醫療評估。')
    with st.sidebar:
        st.button('清除對話', key='clear_chat', on_click=clear_chat, width='stretch')
        render_question_panel()
        with st.expander('關於這個工具'):
            st.write('依據血液檢驗 QA 資料提供一般解釋，回答會附上對應問題與原有參考來源。')
    if not st.session_state['messages']:
        st.subheader('試試這些問題')
        columns = st.columns(2)
        for (i, example) in enumerate(EXAMPLES):
            columns[i % 2].button(example, key=f'example_{i}', on_click=choose_example, args=(example,), width='stretch')
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    prompt = st.chat_input('例如：糖化血色素 HbA1c 反映什麼？', max_chars=1000, key='question')
    example_prompt = st.session_state.pop('pending_question', None)
    question = prompt or example_prompt
    if question and question.strip():
        question = question.strip()
        st.session_state['messages'].append({'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            with st.spinner('正在查詢，首次使用需載入模型…'):
                answer = respond(question)
            st.markdown(answer)
        st.session_state['messages'].append({'role': 'assistant', 'content': answer})
if __name__ == '__main__':
    render_app()
