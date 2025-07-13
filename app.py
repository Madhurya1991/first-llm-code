#   code
#   pip install langchain-google-genai langchain-community faiss-cpu sentence-transformers

from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# --- Setup Gemini LLM ---
google_api_key = input('Enter your API key')
llm = GoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=google_api_key)
response = llm.invoke('Say Hello to workshop participants')
print(response)

# --- 1. Free-form interaction ---
def raw_ai():
  question = input('Ask Gemini anything: ')
  response = llm.invoke(question)
  print(f'\n Gemini: \n {response}')
  
raw_ai()

questions = ['Explain machine learning', 'tell me briefly about machine learning', 'what is machine learning']
for q in questions:
  print(f'\n {q}')
  response = llm.invoke(q)
  print(f'{response[:150]}...')

# --- 2. Structured explanation chain ---
prompt = PromptTemplate(
    input_variables = ['topic'], 
    template=""" Explain {topic} in simple terms with an example:
    
    Format:
    - Keep it under 100 words
    - Use an analogy
    - End with a practical example

    Topic: {topic}
    Explanation: """
    )

filled_prompt = prompt.format(topic='AI agent')
print(filled_prompt)

#first give the prompt
#then connect to llm and get response
#then format the response

chain = prompt | llm | StrOutputParser()

topics = ['blockchain', 'AI agent', 'Internet Model']
for t in topics:
  print(f'\n Explaining: {t}')
  result = chain.invoke({'topic': t})
  print(f'Response: {result}')

company_questions = ['what is your return policy?', 'what are your shipping costs?', 'what are your support hours?']
for q in company_questions:
  response = llm.invoke(q)
  print(f'{q}')
  print(f'{response[:150]}')

# --- 3. Document similarity + RAG setup ---
documents = ['We offer 30-day returns with receipts',
             'Free shipping on orders above 50$',
             'Support available 9 a.m - 5 p.m M-F', 
             'We accept all major credit cards'
             ]
# question = input('give me your question?')


embeddings = HuggingFaceEmbeddings(model_name='all-miniLM-L6-v2')

sample_text = ['return policy', 'refund policy', 'weather forecast']
sample_embeddings = embeddings.embed_documents(sample_text)
print([len(text) for text in sample_text])
print([len(embed) for embed in sample_embeddings])


docs = [Document(page_content=doc) for doc in documents]

vector = FAISS.from_documents(docs, embeddings)


test_query = 'refund policy'
relevant_doc = vector.similarity_search(test_query, k=1)
print(f'{relevant_doc[0].page_content}')

def ask_rag(question):
  relevant_doc = vector.similarity_search(question, k=2)
  context = '\n'.join([doc.page_content for doc in relevant_doc])
  rag_prompt = f"""
  Based on this company information:
  {context}
  Question: {question}
  Please 
  - provide helpful answer
  - first greet customer
  Answer:
  """
  res = llm.invoke(rag_prompt)
  return res

question = 'whats your return ploicy?'
print(ask_rag(question))

