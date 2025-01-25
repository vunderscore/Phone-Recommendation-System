# Phone-Recommendation-System
Phone recommendation application using google gemini LLM and Langchain

This is a phone recommendation AI model using Google Gemini LLM (model: gemini-pro). I have used Langchain to integrate the LLM, rompt template
and database. 

The vector embeddings are done with the help of google genAI embeddings. The vector database used for this application is the FAISS vector database

Check the notebook(ipynb) file to get an understanding of how the LLM integration works.It is not necessary to run the ipynb file though.

When running the application, save all the files to desired directory and use streamlit commands to open the application on your browser.
use either 'streamlit run main.py' or 'python -m streamlit run main.py' if you face issues with the first command.

Once the applicatiion is running click on 'create/update database' to create the initial vector database. Once that is done, then you can start asking
questions to the chatbot. 

Please know that this is a proof of concept application and a lot more tuning and designing is needed before deployment.

Also please check the prompt template in 'langchain_compenent' to get an understanding of the prompt. Also make any necessary changes to tune it to your
liking.

NOTE: please use the your own gemini api-key instead of <your-gemini-api-key> in notebook file and .env file.
