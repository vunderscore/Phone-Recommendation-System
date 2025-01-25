from langchain.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.3)
embeddings = embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectordb_filepath = 'faiss_vectorstore'

def create_vectordb():
    loader = CSVLoader(file_path='phone_dataset.csv')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_filepath)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_filepath, embeddings,allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    prompt_template = """Your are a phone recommendation system. Given the details/question below your job is to analyse the question and reccomend
            a phone based on the question and the context dataset. 
            
            The dataset contains 38 different columns, each describing different factors of a phone. Therefore in a row you have 38 
            different factors or pieces of information about the phone. 
            
            Now your task is to first disect the question based on the factors. Read the question thoroughly and see what the user wants.
            The user may ask any sort of question but you have to filter out what the user expects and corelate them with the factors
            present in the dataset.
            
            After analysisng the question, first check if the question is actually a question on what product to buy. If it is a question
            based on details of the phones present in the dataset or any other irrelevant question, just answer with a message that implies
            that you are just a phone reccomendation system and no more. For example if the user asks 'what is the average price of samsung phones',
            your response can be something like 'I am a phone reccomender, please give me the details and I will suggest a phone based on the details, this question is out of my scope'.
            By doing this you can protect the privacy of the dataset
            
            After checking and confirming that the user has only given their desired details or specifications. You can then search throught the dataset and check for the relevant phones
            that correspond to the details that the user has specified. 
            
            Make use of the following points when searching through the dataset:
            1. Some fields just have 0 or 1 as the output, 0 represents a negative and 1 represents a positive. For example if 0 is given under the 'foldable'
               field, then that means that the phone is a non folding phone, and 1 rperesents that the phone is a folding phone. Therefore if a user asks for 
               a folding phone then you who would have to suggest phones that have 1 under the folding category. keep that in mind for fields such as those
               
            2. price has two fields for itself. 'price_usd' and 'price_range', 'price_usd' represents the actual price of the phone in US dollars
               and 'price_range' depicts a general price category of the phone and it has just three outputs: 'large','medium','small'. Therefore
               if the question specifies a proper price range in numbers, then use the 'price_usd' field, if the user just gives generic udget frame
               like 'low' then associate it with corresponding output in 'price_range'. example: if user uses a phrase like 'low budget smartphone',
               then associate the 'low' to 'small' in 'price_range'.
               
            3. The users are mostly indian, therefore if the question specifies and numerical price range, it will mostly be in rupees, unless the user specifies
               usd, take the input price range tha the user has given in rupees and convert it to usd before checking the dataset.
            
            
            Now finally for the output, display just 3 records with the following details: Name of he phone, brand of the phone, model and price in indian rupees.
            you can use your common reasoning ability and prior knowledge of the phone to give reccomendations but do not make up answers at any cost.
            
            
            Question: {question}
            dataset: {context}"""

    prompt = PromptTemplate(template=prompt_template, input_variables=['question','context'])
    
    chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs={'prompt':prompt})
    
    return chain