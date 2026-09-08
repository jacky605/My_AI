import streamlit as st
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 頁面配置
st.set_page_config(page_title="AI  - 混合 RAG 助手", page_icon="💊")
st.title("💊 本地私有 + 聯網搜尋 混合 RAG 助手")
st.caption("優先檢索本地 ChromaDB 知識庫，查無資料時自動連網搜尋")

# 初始化工具與模型
@st.cache_resource
def init_components():
    # 1. 本地私有資料
    documents = [
        "AI 的內部數據庫系統於 2026 年升級至 FHIR R4 標準。",
        "專案管理員為 Kwok，負責監督 RAG 檢索系統與本地 OLMo 模型的部署。",
        "系統預設的數據切分大小 (Chunk Size) 為 500 字元，重疊 (Overlap) 為 50 字元。",
        "系統的緊急聯絡電話為 0800-123-456，服務時間為週一至週五 09:00-18:00。",
    ]
    
    # 2. 本地 ChromaDB 向量庫
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_texts(texts=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 3. LLM 模型
    llm = ChatOllama(model="olmo2", temperature=0.1)
    
    # 4. 聯網搜尋工具
    web_search = DuckDuckGoSearchRun()
    
    return retriever, llm, web_search

retriever, llm, web_search = init_components()

# Prompt 模版
template = """請根據以下提供資料回答問題：

【參考資料】
{context}

【問題】
{question}

請簡潔直接地回答問題。
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

# 初始化歷史訊息
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是混合 RAG 助手。你可以問我任何問題，或是任何最新的網路資訊！"}
    ]

# 渲染歷史對話
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 使用者輸入
if prompt_text := st.chat_input("請輸入你的問題..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("思考與檢索中..."):
            # Step 1: 試圖從本地向量庫檢索
            local_docs = retriever.invoke(prompt_text)
            
            # 使用 LLM 快速判斷本地文件是否足夠回答問題
            local_context = "\n".join([doc.page_content for doc in local_docs])
            
            # 判斷本地是否有相關內容（如果本地檢索結果包含關鍵資料）
            # 這裡簡單判斷：若問題關鍵詞包含在本地資料中，或者直接讓 LLM 回答
            # 為了確保精準度，我們讓系統先判斷本地資料是否足夠
            is_local_sufficient = False
            
            # 進行雙軌檢索決策
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

            # Step 2: 最終生成回答
            response = chain.invoke({"context": final_context, "question": prompt_text})
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})