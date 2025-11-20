import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Assuming these exist in your project structure
from src.configs.defaults import *
from src.helpers.pdf_processor import PDFProcessor
from src.helpers.vector_store_manager import VectorStoreManager

st.set_page_config(page_title="PDF RAG", layout="wide")

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
def format_docs(docs):
    """Helper function to format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag_chain(vectorstore):
    """Initialize RAG chain using manual LCEL construction"""
    # LLM Initialization
    llm = Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )
    
    # Prompt Template
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    The context may include tables with structured data. Pay special attention to table data when answering questions about specific values, numbers, or statistics.
    
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    
    # Manual LCEL Chain Construction
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def main():
    st.title("PDF RAG System")
    st.markdown("Upload a PDF with tables and ask questions about its content")
    
    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("Document Upload")
        
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'], help="Upload a PDF containing tables")
        
        if uploaded_file is not None:
            # Save uploaded file
            upload_dir = DATA_DIR / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            pdf_path = upload_dir / uploaded_file.name
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Uploaded: {uploaded_file.name}")
            
            # Process button
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Process PDF
                        processor = PDFProcessor(
                            chunk_size=CHUNK_SIZE,
                            chunk_overlap=CHUNK_OVERLAP
                        )
                        documents = processor.process_pdf(str(pdf_path))
                        
                        # Create vector store
                        vs_manager = VectorStoreManager(
                            collection_name=COLLECTION_NAME,
                            model_name=EMBEDDING_MODEL, 
                            ollama_base_url=OLLAMA_BASE_URL
                        )
                        vectorstore = vs_manager.create_vectorstore(documents)
                        
                        st.session_state.vectorstore_ready = True
                        st.session_state.vs_manager = vs_manager
                        st.session_state.pdf_name = uploaded_file.name
                        
                        st.success(f"Processed {len(documents)} chunks!")
                        
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
        
        # Display current status
        st.divider()
        if st.session_state.vectorstore_ready:
            st.success(f"Ready: {st.session_state.get('pdf_name', 'Unknown')}")
        else:
            st.info("Please upload and process a PDF")
        
        # Settings
        st.divider()
        st.subheader("Configurations")
        st.text(f"Model: {MODEL_NAME}")
        st.text(f"Chunks: {CHUNK_SIZE}")
        st.text(f"Top K: {TOP_K_RESULTS}")
    
    # Main chat interface
    if st.session_state.vectorstore_ready:
        st.subheader("Ask Questions")
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the contents of this document?",
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            ask_button = st.button("Ask", type="primary")
        
        with col2:
            clear_button = st.button("Clear History")  # Assign to variable
        
        # Handle clear button outside the column
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
            
        # Process query
        if ask_button and query:
            with st.spinner("Thinking..."):
                try:
                    # Get vectorstore and create chain
                    vectorstore = st.session_state.vs_manager.get_vectorstore()
                    qa_chain, retriever = initialize_rag_chain(vectorstore)
                    
                    # Get relevant documents for display
                    source_docs = retriever.invoke(query)
                    
                    # Invoke the chain
                    answer = qa_chain.invoke(query)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': answer,
                        'sources': source_docs
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display chat history
        if st.session_state.chat_history:
            st.divider()
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    st.markdown(f"**Q{len(st.session_state.chat_history) - i}:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer']}")
                    
                    # Show sources in expander
                    with st.expander("View Sources"):
                        for j, doc in enumerate(chat['sources'], 1):
                            st.markdown(f"**Source {j}** (Page {doc.metadata.get('page', 'N/A')}, Type: {doc.metadata.get('type', 'N/A')})")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.divider()
                    
                    st.markdown("---")
    
    else:
        st.info("Upload and process a PDF to get started!")
    

if __name__ == "__main__":
    main()