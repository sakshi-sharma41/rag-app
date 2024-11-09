import streamlit as st
import logging
# import ollama
from typing import List, Tuple, Dict, Any, Optional

# import user-defined modules for the RAG application
from chat_model import process_question
from vectordb import create_vector_db, delete_vector_db
from pdf_to_text import extract_all_pages_as_images

# On non-Windows systems, switch to an alternative SQLite package for compatibility
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Streamlit page configuration
st.set_page_config(
    page_title="Insight Matrix RAG",          # Sets page title
    page_icon="💼",                           # Sets favicon
    layout="wide",                            # Sets layout to wide-screen
    initial_sidebar_state="collapsed",        # Hides sidebar on page load
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # Logger for error and info messages

@st.cache_resource(show_spinner=True)
def extract_model_names():
    """
    Function to extract model names. For now, it returns a static model name
    until the model list extraction from Ollama (or other source) is configured.
    """
    # Placeholder for model extraction, currently returns "gamma_model"
    model_names = "gamma_model"
    return model_names

def main() -> None:
    """
    Main function to run the Streamlit application.
    
    Sets up the user interface, handles file uploads, and processes user queries.
    """
    # Display application title
    st.subheader("IntelliQuest 📈", divider="gray", anchor=False)

    available_models = extract_model_names()  # Get available model names

    # Create layout with two columns
    col2, col1 = st.columns([2, 1.5])

    # Initialize session state variables if not already present
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    # Dropdown to select an available model
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )

    # File uploader for the PDF file, only accepting single files of PDF format
    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", type="pdf", accept_multiple_files=False
    )

    # Process file upload if a file is uploaded
    if file_upload:
        st.session_state["file_upload"] = file_upload  # Save file in session state

        # Create vector database if not already created
        if st.session_state["vector_db"] is None:
            st.session_state["vector_db"] = create_vector_db(file_upload)

        # Extract pages as images from the uploaded PDF
        pdf_pages = extract_all_pages_as_images(file_upload)
        st.session_state["pdf_pages"] = pdf_pages

        # Slider to adjust zoom level of displayed PDF images
        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50
        )

        # Display each PDF page image within a container
        with col1:
            with st.container(height=410, border=True):
                for page_image in pdf_pages:
                    st.image(page_image, width=zoom_level)

    # Button to delete vector database collection
    delete_collection = col1.button("⚠️ Delete collection", type="secondary")
    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    # Chat input and message display container
    with col2:
        message_container = st.container(height=500, border=True)

        # Display each chat message stored in session state
        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Input box for user prompt
        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                # Add user message to session state and display it
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                # Generate and display assistant's response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                # Store assistant's response in session state
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")

        else:
            # Display warning if no PDF file is uploaded when a chat is attempted
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")

# Entry point of the application
if __name__ == "__main__":
    main()
