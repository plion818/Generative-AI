# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 18:36:56 2025

@author: ricky
"""

import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

# 呼叫函式並傳入 PDF 路徑
result = extract_text_from_pdf("book.pdf")
# 顯示前200字
print(result[:200])


#文本切割成段落
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
texts = splitter.split_text(result)


#文本向量化
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
# 把文本清單轉換成向量
vectors = encoder.encode(texts, show_progress_bar=True)




#建立FAISS索引
import faiss
import numpy as np

# 建立索引
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
# 加入向量
index.add(np.array(vectors))
# 把 texts 和對應索引連起來
id_to_text = {i: text for i, text in enumerate(texts)}




#單題範例問答（先測試一題）
query = "請問 LLaMA 是什麼？"
query_vec = encoder.encode([query])
# 查前 5 名
D, I = index.search(np.array(query_vec), k=5)
# 取回文字段落
retrieved_context = "\n".join([id_to_text[i] for i in I[0] if i >= 0])






#加入載入 Open LLaMA 模型
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "openlm-research/open_llama_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")






# 組 prompt，讓 Open LLaMA 根據檢索到的段落來回答問題
prompt_template = f"""
你是一個中文助理，請根據以下內容回答問題。

文件內容：
{retrieved_context}

問題：
{query}

請用簡單中文作答：
"""
# 編碼 prompt
inputs = tokenizer(prompt_template, return_tensors="pt").to("cuda")

# 產生回答
outputs = model.generate(**inputs, max_new_tokens=512)

# 解碼結果
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 顯示回答
print("\n回答：")
print(answer)





"""回答問題"""

import json

# 讀取 JSON 檔
with open("queries.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# 顯示所有問題
for q in questions:
    print(q["id"], q["question"])


results = []

for q in questions:
    query = q["question"]

    # 向量化 query
    query_vec = encoder.encode([query])

    # 查前 5 名
    D, I = index.search(np.array(query_vec), k=5)

    # 把檢索到的段落組起來
    retrieved_context = "\n".join([id_to_text[i] for i in I[0]])

    # 組 Prompt
    prompt_template = f"""
    以下是一些文件內容：
    {retrieved_context}

    請根據上面的文件，回答以下問題：
    問題：{query}

    請用中文簡單明確作答：
    """

    # 丟給 LLaMA
    inputs = tokenizer(prompt_template, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 存結果
    results.append({
        "id": q["id"],
        "question": query,
        "answer": answer
    })

# 最後寫回成 JSON
with open("answers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)







