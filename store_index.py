# from src.helper import load_pdf, text_split, download_hugging_face_embeddings
# from langchain_community.vectorstores import Pinecone
# import pinecone
# from dotenv import load_dotenv
# import os

# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# # print(PINECONE_API_KEY)
# # print(PINECONE_API_ENV)

# extracted_data = load_pdf("data/")
# text_chunks = text_split(extracted_data)
# embeddings = download_hugging_face_embeddings()


# #Initializing the Pinecone
# pinecone.init(api_key=PINECONE_API_KEY,
#               environment=PINECONE_API_ENV)


# index_name="medical-bot"

# #Creating Embeddings for Each of The Text Chunks & storing
# docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)


from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Load and split documents
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# ✅ Initialize Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Define index name
index_name = "medical-bot"

# Optional: create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # for MiniLM model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_API_ENV  # example: 'us-west-1'
        )
    )

# ✅ Use Langchain to store the data
docsearch = LangchainPinecone.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name
)
