from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMMA_KEYGOOGLE_API_KEY")

# Logging configuration for debug information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # Logger for logging events in the function

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    # Log the question and selected model for debugging
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize the language model with specific parameters
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",         # Specifies the model name
        temperature=0,                  # Controls creativity/variation in responses
        max_tokens=None,                # Sets the token limit, None means default
        timeout=None,                   # Sets response timeout, None means default
        max_retries=2                   # Number of times to retry on failure
        # other params...
    )
    
    # Prompt template to generate alternative queries for better document retrieval
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Initialize the multi-query retriever with vector database retriever, language model, and prompt
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    # Template for the main query prompt that combines context and question
    template = """
        You are an AI language model. Answer the question using the following context and your own knowledge:
        Context: {context}
        Question: {question}

        If the context does not provide sufficient information, use your own knowledge to answer the question. However, if you still don't know the answer, simply say that you don't know.
        """

    # Create the chat prompt from the template
    prompt = ChatPromptTemplate.from_template(template)

    # Define the process chain:
    # 1. Pass "context" through retriever
    # 2. Pass "question" as is through RunnablePassthrough
    # 3. Format the prompt with context and question
    # 4. Pass through language model (llm) to generate response
    # 5. Parse response with StrOutputParser to get a clean string
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain with the question and get the response
    response = chain.invoke(question)
    logger.info("Question processed and response generated")  # Log success

    return response
