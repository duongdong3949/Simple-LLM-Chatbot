from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
# from langchain.retrievers.multi_query import MultiQueryRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig 
from operator import itemgetter
import torch
import logging

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
print(f"CUDA available: {torch.cuda.is_available()}")
use_cuda = torch.cuda.is_available()
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cuda" if use_cuda else "cpu"}
)
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16 
# )

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    # quantization_config=quantization_config,
    device_map="auto",
    offload_buffers=True
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

# retriever = MultiQueryRetriever.from_llm(
#     retriever=base_retriever,
#     llm=llm
# )

memory = ConversationBufferMemory(
    memory_key='chat_history',
    input_key='question',
    output_key='output',
    return_messages=False
)

prompt_template = """
<|im_start|>system
Bạn là một trợ lý nhà hàng rất thân thiện và am hiểu thực đơn.
- Luôn trả lời một cách vui vẻ, lịch sự.
- Chỉ sử dụng thông tin từ "Thông tin thực đơn" được cung cấp. Không tự bịa ra thông tin.
- Nếu thông tin không có, hãy trả lời một cách khéo léo, ví dụ: "Dạ, em chưa tìm thấy thông tin về món này trong thực đơn ạ. Anh/chị có muốn tham khảo món khác không ạ?"
- Khi liệt kê giá, luôn ghi kèm đơn vị "đồng".
- Nhiệm vụ tối quan trọng: Bạn chỉ được phép trả lời bằng tiếng Việt Nam thuần túy.
- Tuyệt đối không được sử dụng tiếng Trung Quốc trong câu trả lời.
<|im_end|>
<|im_start|>user
Lịch sử trò chuyện:
{chat_history}
Thông tin thực đơn:
---------------------
{context}
---------------------
Hãy trả lời câu hỏi:
{question}
<|im_end|>
<|im_start|>assistant
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"]
)

def format_docs(docs):
    return '\n'.join([doc.page_content for doc in docs])

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),  
        "chat_history": RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
    }
    | prompt
    | llm
    | StrOutputParser()
)

def get_response(query):
    response = rag_chain.invoke(query)
    response = response.split("<|im_start|>assistant\n")[1]
    memory.save_context({"question": query}, {"output": response})
    return response
