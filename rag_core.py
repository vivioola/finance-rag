import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

def build_index(pdf_path="report.pdf"):
    print("正在加载 PDF...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"共加载 {len(docs)} 页")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"共切分 {len(chunks)} 个片段")

    print("正在建立向量索引（分批处理）...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = None
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"处理第 {i+1}-{min(i+batch_size, len(chunks))} 片段...")
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        time.sleep(10)

    vectorstore.save_local("faiss_index")
    print("向量索引已保存")
    return vectorstore

def ask(question, vectorstore):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    relevant_docs = retriever.invoke(question)

    # 构建带页码的 context
    context_parts = []
    sources = []
    for doc in relevant_docs:
        page = doc.metadata.get("page", "?") + 1  # pymupdf 页码从0开始
        context_parts.append(f"[第{page}页]\n{doc.page_content}")
        sources.append(f"第{page}页")

    context = "\n\n".join(context_parts)

    prompt = f"""你是一个金融研报分析助手。请根据以下报告内容回答问题。
回答时必须在每个要点后注明来源页码，格式为（第X页）。

报告内容：
{context}

问题：{question}

请用中文回答，并在关键信息后标注页码来源。"""

    response = llm.invoke(prompt)
    return response.content, list(set(sources))

if __name__ == "__main__":
    # 如果已有索引就直接加载，不重新建
    import os
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if os.path.exists("faiss_index"):
        print("加载已有索引...")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = build_index()

    question = "这份报告的主要结论是什么？"
    answer, sources = ask(question, vectorstore)
    print(f"\n问题：{question}")
    print(f"回答：\n{answer}")
    print(f"\n来源：{', '.join(sources)}")
