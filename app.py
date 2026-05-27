import time
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="金融研报问答助手", page_icon="📊", layout="wide")
st.title("📊 金融研报智能问答助手")
st.caption("上传研报 PDF，用 AI 提问获取关键信息")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

@st.cache_resource
def load_existing_index():
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

def build_index(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectorstore = None
    batch_size = 20
    progress = st.progress(0, text="正在建立索引...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        progress.progress(min((i+batch_size)/len(chunks), 1.0), text=f"处理第 {i+1}-{min(i+batch_size, len(chunks))}/{len(chunks)} 片段...")
        time.sleep(10)
    vectorstore.save_local("faiss_index")
    progress.empty()
    return vectorstore

def ask(question, vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)
    context_parts = []
    sources = []
    for doc in relevant_docs:
        page = doc.metadata.get("page", 0) + 1
        context_parts.append(f"[第{page}页]\n{doc.page_content}")
        sources.append(page)
    context = "\n\n".join(context_parts)
    prompt = f"""你是一个研报分析助手。根据以下报告内容回答问题，在关键信息后注明来源页码（第X页）。

报告内容：
{context}

问题：{question}

用中文回答，标注页码来源。"""
    response = llm.invoke(prompt)
    return response.content, sorted(set(sources))

# 侧边栏上传
with st.sidebar:
    st.header("📁 上传研报")
    uploaded_file = st.file_uploader("选择 PDF 文件", type="pdf")
    if uploaded_file:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        if st.button("建立索引", type="primary"):
            with st.spinner("处理中，约需 1-2 分钟..."):
                st.session_state.vectorstore = build_index("uploaded.pdf")
            st.success("索引建立完成！")

    st.divider()
    st.header("📌 使用已有研报")
    if st.button("加载默认研报"):
        vs = load_existing_index()
        if vs:
            st.session_state.vectorstore = vs
            st.success("已加载！")
        else:
            st.error("未找到已有索引")

# 主界面
if "vectorstore" not in st.session_state:
    st.info("👈 请先在左侧加载研报")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                st.caption(f"📄 来源页码：{', '.join([f'第{p}页' for p in msg['sources']])}")

    if question := st.chat_input("请输入你的问题，例如：这份报告的主要结论是什么？"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                answer, sources = ask(question, st.session_state.vectorstore)
            st.markdown(answer)
            st.caption(f"📄 来源页码：{', '.join([f'第{p}页' for p in sources])}")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
