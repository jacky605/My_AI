from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 1. 準備私有測試資料
documents = [
    "AI Health Pharmacy 的內部數據庫系統於 2026 年升級至 FHIR R4 標準。",
    "專案管理員為 Kwok，負責監督 RAG 檢索系統與本地 OLMo 模型的部署。",
    "系統預設的數據切分大小 (Chunk Size) 為 500 字元，重疊 (Overlap) 為 50 字元。",
    "藥局系統的緊急聯絡電話為 0800-123-456，服務時間為週一至週五 09:00-18:00。",
]

# 2. 初始化 Embedding 模型與向量資料庫
print("正在載入向量資料庫...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_texts(texts=documents, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. 初始化 LLM
llm = ChatOllama(model="olmo2", temperature=0.1)

# 4. 優化後的 Prompt 模版
template = """請根據以下背景資料回答問題：

【背景資料】
{context}

【問題】
{question}

請根據資料內容直接回答，若資料完全未提及才回答「資料未提及」。
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. 構建 RAG 鏈
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 互動式問答迴圈 (Terminal UI)
print("\n" + "=" * 50)
print("🤖 本地 RAG 問答系統已準備就緒！(輸入 'exit' 或 'quit' 即可結束)")
print("=" * 50 + "\n")

while True:
    try:
        # 接收使用者輸入
        user_query = input("使用者提問：").strip()

        # 檢查離開指令
        if user_query.lower() in ["exit", "quit", "exit()", "離去", "退出"]:
            print("系統已結束，再見！")
            break

        # 忽視空白輸入
        if not user_query:
            continue

        # 執行 RAG 檢索與回答
        print("思考中...\n")
        response = rag_chain.invoke(user_query)
        print(f"助手回答：\n{response}\n")
        print("-" * 50)

    except (KeyboardInterrupt, EOFError):
        print("\n系統已結束，再見！")
        break