import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 頁面配置
st.set_page_config(page_title="AI - 混合 RAG 助手", page_icon="💊", layout="wide")
st.title("💊 本地私有 + 聯網搜尋 混合 RAG 助手")
st.caption("優先檢索本地 ChromaDB 知識庫，查無資料時自動連網搜尋（支援變更預覽與安全修改）")

# 1. 初始化資料夾與預設文件
DATA_DIR = os.path.abspath("./data")
os.makedirs(DATA_DIR, exist_ok=True)

default_file = os.path.join(DATA_DIR, "system_info.txt")
if not os.path.exists(default_file):
    with open(default_file, "w", encoding="utf-8") as f:
        f.write(
            "AI 的內部數據庫系統於 2026 年升級至 FHIR R4 標準。\n"
            "專案管理員為 Kwok，負責監督 RAG 檢索系統與本地 OLMo 模型的部署。\n"
            "系統預設的數據切分大小 (Chunk Size) 為 500 字元，重疊 (Overlap) 為 50 字元。\n"
            "系統的緊急聯絡電話為 0800-123-456，服務時間為週一至週五 09:00-18:00。"
        )

# 2. 初始化工具與模型
@st.cache_resource
def init_components():
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(f"【檔案: {filename}】\n" + f.read())
    
    if not documents:
        documents = ["目前本地知識庫無任何資料。"]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_texts(texts=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatOllama(model="olmo2", temperature=0.1)
    web_search = DuckDuckGoSearchRun()
    
    return retriever, llm, web_search

retriever, llm, web_search = init_components()

# 3. 側邊欄：安全修改介面（含有預覽與確認機制）
st.sidebar.header("📁 本地知識庫檔案管理")
file_list = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

if file_list:
    selected_file = st.sidebar.selectbox("選擇欲修改的檔案：", file_list)
    target_filepath = os.path.join(DATA_DIR, selected_file)
    
    with open(target_filepath, "r", encoding="utf-8") as f:
        original_content = f.read()
    
    edited_content = st.sidebar.text_area("編輯檔案內容：", original_content, height=180)
    
    if st.sidebar.button("🔍 預覽變更與確認修改"):
        st.session_state.pending_action = {
            "type": "update",
            "filename": selected_file,
            "filepath": target_filepath,
            "content": edited_content
        }

st.sidebar.markdown("---")
st.sidebar.subheader("➕ 建立新文件")
new_filename = st.sidebar.text_input("新檔案名稱 (例: sop.txt)")
new_file_content = st.sidebar.text_area("新檔案內容", height=100)

if st.sidebar.button("🔍 預覽並建立新檔"):
    if new_filename and new_filename.endswith(".txt"):
        new_filepath = os.path.join(DATA_DIR, new_filename)
        st.session_state.pending_action = {
            "type": "create",
            "filename": new_filename,
            "filepath": new_filepath,
            "content": new_file_content
        }
    else:
        st.sidebar.error("請輸入有效的 .txt 檔名！")

# 4. 變更確認彈出卡片（顯示在對話視窗上方或側邊欄底部）
if st.session_state.pending_action:
    action = st.session_state.pending_action
    st.sidebar.warning("⚠️ 請核對以下變更細節：")
    st.sidebar.markdown(f"**檔案名稱:** `{action['filename']}`")
    st.sidebar.markdown(f"**檔案位置:** `{action['filepath']}`")
    st.sidebar.text_area("預覽即將寫入的內容：", action['content'], height=120, disabled=True)
    
    col_confirm, col_cancel = st.sidebar.columns(2)
    
    if col_confirm.button("✅ 確認無誤，執行"):
        with open(action['filepath'], "w", encoding="utf-8") as f:
            f.write(action['content'])
        st.cache_resource.clear()
        st.session_state.pending_action = None
        st.sidebar.success("變更已成功寫入硬碟，並重新同步向量庫！")
        st.rerun()
        
    if col_cancel.button("❌ 取消"):
        st.session_state.pending_action = None
        st.rerun()

# 5. Prompt 模版與主對話邏輯
template = """請根據以下提供資料回答問題：

【參考資料】
{context}

【問題】
{question}

請簡潔直接地回答問題。
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是混合 RAG 助手。你可以問我任何問題，或是透過左側邊欄安全地預覽與編輯本地知識庫！"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("請輸入你的問題..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("思考與檢索中..."):
            local_docs = retriever.invoke(prompt_text)
            local_context = "\n".join([doc.page_content for doc in local_docs])
            
            check_prompt = f"判斷以下『背景資料』是否包含足夠資訊來回答『問題』。如果包含請只回答 YES，否則只回答 NO。\n背景資料：{local_context}\n問題：{prompt_text}"
            check_res = llm.invoke(check_prompt).content.strip()
            
            if "YES" in check_res.upper():
                st.info("💡 來源：本地私有知識庫 (ChromaDB)")
                final_context = local_context
            else:
                st.warning("🌐 本地資料庫查無相關記錄，已自動升級為聯網搜尋 (DuckDuckGo)")
                try:
                    search_result = web_search.run(prompt_text)
                    final_context = search_result
                except Exception as e:
                    final_context = f"無法進行網路搜尋：{e}"

            response = chain.invoke({"context": final_context, "question": prompt_text})
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})