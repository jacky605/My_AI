# 💊 Hybrid Local RAG Assistant

一個基於 **LangChain**、**ChromaDB** 與 **Ollama (OLMo 2)** 構建的私有化混合式檢索增強生成（Hybrid RAG）系統。

本專案支援**優先檢索本地私有知識庫**（如內部藥局 SOP、醫療數據標準），當本地知識庫查無相關結果時，系統會**自動升級為 DuckDuckGo 聯網即時搜尋**，並透過 Streamlit 提供流暢的 Web UI 對話介面。

---

## 💡 核心優勢與亮點

* 🔒 **數據 100% 本地安全**：採用本地 LLM 與嵌入模型，敏感內部資料不傳送至外部雲端 API。
* 🔀 **Smart Hybrid RAG（雙軌檢索）**：
  * **Level 1 (本地私有庫)**：優先查詢 ChromaDB 向量庫。
  * **Level 2 (聯網備援)**：當本地資訊不足時，自動觸發 DuckDuckGo 搜尋引擎補充最新資訊。
* ⚡ **高效套件管理**：使用超高速 Python 管理器 `uv`，自動處理 `.venv` 隔離與環境鎖定。
* 🖥️ **ChatGPT 風格 Web UI**：基於 Streamlit 實現的互動式對話視窗，具備 Session 對話歷史紀錄與狀態提示。

---

## 🏗️ 系統架構與檢索流程 (Architecture)

```text
               ┌──────────────────────┐
               │    使用者提問 (User)  │
               └──────────┬───────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │   Nomic Embed Text 向量化   │
            └─────────────┬──────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │   檢索 ChromaDB 本地庫   │
             └────────────┬─────────────┘
                          │
                ┌─────────┴─────────┐
                │ 相關資料足夠嗎？  │
                └────┬──────────┬───┘
               YES   │          │  NO
                     ▼          ▼
┌────────────────────────┐  ┌────────────────────────┐
│  💡 使用本地私有 Context │  │  🌐 觸發 DuckDuckGo    │
│  (FHIR R4 / 內部 SOP)  │  │     Web Search 檢索     │
└────────────┬───────────┘  └────────────┬───────────┘
             │                           │
             └────────────┬──────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │    OLMo 2 模型生成回答   │
             └────────────┬──────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Streamlit Web 介面輸出  │
             └──────────────────────────┘

```

---

## 🛠️ 技術棧 (Tech Stack)

* **LLM Engine**: [Ollama](https://ollama.com/) (`olmo2`)
* **Embedding Model**: `nomic-embed-text`
* **Orchestration**: [LangChain](https://www.langchain.com/) (`langchain-chroma`, `langchain-ollama`)
* **Vector Database**: [ChromaDB](https://www.trychroma.com/)
* **Web Search**: DuckDuckGo Search API (`duckduckgo-search`, `ddgs`)
* **Web UI Framework**: [Streamlit](https://streamlit.io/)
* **Environment & Package Manager**: [uv](https://github.com/astral-sh/uv)

---

## 📋 環境需求 (Prerequisites)

在開始安裝前，請確保你的電腦滿足以下環境需求：

1. **作業系統**: Windows 10/11, macOS, 或 Linux
2. **Python**: `>= 3.10`
3. **uv** (Python 超高速包管理器)
```powershell
# Windows PowerShell 安裝 uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


4. **Ollama** (本地 LLM 伺服器)
* 請至 [Ollama 官網](https://ollama.com/) 下載並安裝背景服務。



---

## 🚀 快速開始指南 (Quick Start)

### 步驟 1：下載 Ollama 本地模型

請開啟 Terminal / PowerShell，執行以下命令下載 LLM 與 Embedding 模型：

```bash
# 下載預設 LLM (OLMo 2)
ollama pull olmo2

# 下載文本向量模型 (Nomic Embed)
ollama pull nomic-embed-text

```

### 步驟 2：複製專案與安裝依賴

```bash
# 克隆專案到本地
git clone https://github.com/your-username/my-rag-project.git
cd my-rag-project

# 使用 uv 一鍵建立虛擬環境並同步依賴項
uv sync

```

*(可選)* 如果你想手動在自己的環境安裝所有依賴套件：

```bash
uv add langchain-chroma langchain-ollama langchain-community streamlit duckduckgo-search ddgs

```

---

## 💻 運行系統 (How to Run)

### 1. 啟動 Streamlit 視覺化網頁介面 (主程式)

```bash
uv run streamlit run app.py

```

執行後，瀏覽器將自動打開發佈頁面：`http://localhost:8501`。

### 2. 啟動 Terminal 命令行測試模式

如果你只需要在命令列中進行快速問答測試：

```bash
uv run main.py

```

---

## 📂 專案檔案架構 (Directory Structure)

```text
my-rag-project/
├── app.py              # Streamlit Web UI 主程式 (包含雙軌 RAG 邏輯)
├── main.py             # CLI 終端機測試腳本
├── pyproject.toml      # uv 專案依賴配置文件
├── uv.lock             # uv 依賴精確鎖定檔
├── README.md           # 專案詳細說明文件
└── .gitignore          # Git 忽略檔案設定

```

---

## 📖 如何修改為你自己的私有資料庫

目前 `app.py` 內使用陣列作為測試資料。若要換成你自己的知識庫（如 PDF/TXT 文件），請修改 `app.py` 中的 `init_components` 函數：

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 讀取本地 ./data 資料夾下的所有 PDF
loader = PyPDFDirectoryLoader("./data")
raw_docs = loader.load()

# 切分文件
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(raw_docs)

# 建立 Chroma 向量資料庫
vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

```

---

## ❓ 常見問題排查 (Troubleshooting)

* **Q: 啟動時提示 `ImportError: Could not import ddgs python package`**
* **解決方案**：執行 `uv add ddgs duckduckgo-search` 補全搜尋工具套件。


* **Q: 提示 `DeprecationWarning: langchain-community is being sunset`**
* **解決方案**：本專案已全面升級至 `langchain-chroma` 獨立模組，請確保使用 `uv sync` 更新至最新依賴。


* **Q: 網頁一直顯示「思考與檢索中...」卡住？**
* **解決方案**：請確認背景的 Ollama 服務正在正常運行（可在 Terminal 執行 `ollama list` 檢查模型是否存在）。



---

## 📄 授權條款 (License)

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
