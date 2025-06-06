# -*- coding: utf-8 -*-
"""
Knowledge QA script using PDF text extraction, FAISS, and LLaMA.
Processes questions from a JSON file and generates answers based on the PDF content.
"""

import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

# 呼叫函式並傳入 PDF 路徑
# result = extract_text_from_pdf("book.pdf")
# 顯示前200字
# print(result[:200])


#文本切割成段落
def split_text(full_text: str) -> list[str]:
    """
    Splits the text into chunks of approximately 1500 characters with an overlap of 100 characters.
    """
    chunks = []
    start = 0
    text_len = len(full_text)
    while start < text_len:
        end = start + 1500
        chunk = full_text[start:min(end, text_len)]
        chunks.append(chunk)

        # Calculate start of the next chunk with overlap
        start += (1500 - 100)

        # If the next chunk would be very small (e.g., less than overlap), or if we are near the end
        if start + 100 >= text_len: # If remaining part is less than overlap, just take all of it
            if start < text_len: # If there's any text left
                 # Check if the last chunk is substantially different from the current one
                 # to avoid near-duplicate last chunks.
                if len(full_text[start:]) > 100 : # Only add if the remaining part is of reasonable size
                    chunks.append(full_text[start:])
            break

    # Post-processing: remove very small chunks if any were accidentally created, except if it's the only chunk
    if len(chunks) > 1:
        chunks = [c for c in chunks if len(c.strip()) > 50] # Arbitrary small threshold, e.g. 50 chars

    # Ensure there's at least one chunk if the original text was not empty
    if not chunks and full_text:
        chunks.append(full_text)

    return chunks

# texts = splitter.split_text(result)


#文本向量化
# from sentence_transformers import SentenceTransformer # Moved to top
# encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5") # Instantiated in main


#建立FAISS索引
# import faiss # Moved to top
# import numpy as np # Moved to top

def create_vector_db(texts: list[str], encoder_model: SentenceTransformer):
    """
    Creates a FAISS index from a list of texts using the provided encoder model.
    """
    print("Encoding texts for FAISS...")
    vectors = encoder_model.encode(texts, show_progress_bar=True)
    if not vectors.any(): # Check if vectors is empty or all zeros
        raise ValueError("Text encoding resulted in empty vectors. Cannot build FAISS index.")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors, dtype=np.float32)) # Ensure correct dtype for FAISS
    id_to_text = {i: text for i, text in enumerate(texts)}
    return index, id_to_text


# #單題範例問答（先測試一題）
# query = "請問 LLaMA 是什麼？"
# query_vec = encoder.encode([query])
# # 查前 5 名
# D, I = index.search(np.array(query_vec), k=5)
# # 取回文字段落
# retrieved_context = "\n".join([id_to_text[i] for i in I[0] if i >= 0])


# #加入載入 Open LLaMA 模型
# from transformers import AutoTokenizer, AutoModelForCausalLM # Moved to top
# model_name = "openlm-research/open_llama_7b" # Defined in main
# tokenizer = AutoTokenizer.from_pretrained(model_name) # Instantiated in main
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") # Instantiated in main


# # 組 prompt，讓 Open LLaMA 根據檢索到的段落來回答問題
# prompt_template = f"""
# 你是一個中文助理，請根據以下內容回答問題。

# 文件內容：
# {retrieved_context}

# 問題：
# {query}

# 請用簡單中文作答：
# """
# # 編碼 prompt
# inputs = tokenizer(prompt_template, return_tensors="pt").to("cuda")

# # 產生回答
# outputs = model.generate(**inputs, max_new_tokens=512)

# # 解碼結果
# answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # 顯示回答
# print("\n回答：")
# print(answer)




"""回答問題"""

# import json # Moved to top

def get_answer(query: str, index: faiss.Index, id_to_text_map: dict,
               encoder_model: SentenceTransformer,
               llm_model: AutoModelForCausalLM, llm_tokenizer: AutoTokenizer) -> str:
    """
    Retrieves relevant context from FAISS and generates an answer using the LLM.
    """
    print(f"Vectorizing query: '{query}'")
    query_vec = encoder_model.encode([query], convert_to_tensor=False, show_progress_bar=False) # FAISS expects numpy array

    print("Searching FAISS index...")
    try:
        D, I = index.search(np.array(query_vec, dtype=np.float32), k=5) # Ensure correct dtype
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return "FAISS 索引搜索時發生錯誤。"

    retrieved_indices = [i for i in I[0] if i >= 0 and i < len(id_to_text_map)] # Filter out invalid indices
    if not retrieved_indices:
        print("No relevant documents found in FAISS index for the query.")
        return "文件中未找到相關資訊"

    retrieved_context = "\n---\n".join([id_to_text_map[i] for i in retrieved_indices])
    print(f"Retrieved {len(retrieved_indices)} chunks from FAISS.")

    prompt = f"""請根據文件語意，若文件中無相關資訊，請明確回覆「文件中未找到相關資訊」。

文件內容：
{retrieved_context}

問題：
{query}

請用簡單中文作答：
"""
    print("Generating answer with LLM...")
    try:
        inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(llm_model.device) # Added truncation and max_length
        outputs = llm_model.generate(**inputs, max_new_tokens=512, pad_token_id=llm_tokenizer.eos_token_id) # Use eos_token_id for pad_token_id if pad_token is not set
        answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the answer if the model includes it
        # This can happen if the model is not strictly an instruction-following model
        # or if the prompt is too long / complex.
        # A simple way is to check if the answer starts with the prompt.
        if answer.strip().startswith(prompt.strip()): # Check with strip to handle potential leading/trailing spaces
             answer = answer[len(prompt):].strip()
        elif query in answer: # A less strict check, try to remove the question part
            # This is a heuristic. If the model's output is structured, it might be "Answer: actual answer"
            # For now, we'll just remove the first occurrence of the query if the prompt isn't there.
            # More sophisticated parsing might be needed depending on the model's typical output.
            # Find the part of the answer that starts after the query
            query_pos = answer.find(query)
            if query_pos != -1:
                # Try to find a natural break after the query (e.g., newline or "作答：")
                split_markers = ["作答：", "\n", "回答："]
                actual_answer_start = -1
                for marker in split_markers:
                    marker_pos = answer.find(marker, query_pos + len(query))
                    if marker_pos != -1:
                        actual_answer_start = marker_pos + len(marker)
                        break
                if actual_answer_start != -1:
                    answer = answer[actual_answer_start:].strip()
                else: # Fallback: take text after the query
                    answer = answer[query_pos + len(query):].strip()
                    # If the answer starts with a colon or similar punctuation, remove it.
                    if answer.startswith(":") or answer.startswith("："):
                        answer = answer[1:].strip()

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "生成回答時發生錯誤。"

    # Final check if the answer is empty or just placeholder after stripping prompt
    if not answer.strip() or answer.strip() == "請用簡單中文作答：" or answer.strip() == "文件中未找到相關資訊。" and not retrieved_indices :
        return "文件中未找到相關資訊"

    return answer


def main():
    pdf_path = "book.pdf"
    queries_path = "queries.json"
    answers_path = "answers.json"
    encoder_model_name = "BAAI/bge-small-zh-v1.5"
    llm_model_name = "openlm-research/open_llama_7b" # Consider a smaller model if 7B is too large for environment

    # Extract text from PDF
    print(f"Extracting text from {pdf_path}...")
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text.strip():
        print(f"No text extracted from {pdf_path}. Exiting.")
        # Create empty answers.json or handle as error
        with open(answers_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return
    print(f"Extracted {len(full_text)} characters.")

    # Split text into chunks
    print("Splitting text into chunks...")
    text_chunks = split_text(full_text)
    if not text_chunks:
        print("Text splitting resulted in no chunks. Exiting.")
        # Create empty answers.json or handle as error
        with open(answers_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return
    print(f"Split into {len(text_chunks)} chunks.")

    # Load encoder model
    print(f"Loading encoder model: {encoder_model_name}...")
    try:
        encoder_model = SentenceTransformer(encoder_model_name)
    except Exception as e:
        print(f"Error loading encoder model: {e}. Please ensure it is installed or accessible.")
        return
    print("Encoder model loaded.")

    # Create FAISS vector database
    print("Creating FAISS vector database...")
    try:
        vector_index, id_to_text_mapping = create_vector_db(text_chunks, encoder_model)
    except ValueError as e:
        print(f"Error creating FAISS DB: {e}")
        return
    except Exception as e: # Catch other potential FAISS errors
        print(f"An unexpected error occurred while creating FAISS DB: {e}")
        return
    print("FAISS index created.")

    # Load LLM model and tokenizer
    print(f"Loading LLM model: {llm_model_name}...")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        # Add pad token if it's missing (common for some LLaMA models)
        if llm_tokenizer.pad_token is None:
            # Using eos_token as pad_token if pad_token is not set
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
            print(f"Tokenizer pad_token not set, using eos_token ({llm_tokenizer.eos_token}) as pad_token.")

        # Ensure offload folder exists
        import os
        offload_folder = os.path.join(os.path.dirname(__file__), "offload")
        os.makedirs(offload_folder, exist_ok=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            offload_folder=offload_folder
        )
        # No need to resize token embeddings if only setting pad_token to eos_token,
        # but if new special tokens were added and vocabulary expanded, it would be necessary:
        # llm_model.resize_token_embeddings(len(llm_tokenizer))
    except Exception as e:
        print(f"Error loading LLM model or tokenizer: {e}. Please ensure model name is correct and resources are available.")
        return
    print("LLM model loaded.")

    # Read queries
    print(f"Loading queries from {queries_path}...")
    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            queries_data = json.load(f)
    except FileNotFoundError:
        print(f"Queries file not found: {queries_path}. Exiting.")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {queries_path}. Exiting.")
        return
    print(f"Loaded {len(queries_data)} queries.")

    results = []
    print("Processing queries...")
    if not isinstance(queries_data, list):
        print("Queries data is not a list. Please check the format of queries.json.")
        return

    for q_data in queries_data:
        if not isinstance(q_data, dict):
            print(f"Skipping invalid query data: {q_data}")
            continue
        # Accept both 'id' and 'query_id' as valid keys
        query_id = q_data.get("id") or q_data.get("query_id")
        if query_id is None or "question" not in q_data:
            print(f"Skipping invalid query data: {q_data}")
            continue

        question_text = q_data["question"]
        print(f"Processing query ID: {query_id}, Question: {question_text}")

        answer_text = get_answer(question_text, vector_index, id_to_text_mapping, encoder_model, llm_model, llm_tokenizer)
        print(f"  Raw Answer from LLM: {answer_text}") # Log raw answer for debugging

        results.append({
            "query_id": query_id,
            "question": question_text,
            "answer": answer_text
        })

    # Save results
    print(f"Saving answers to {answers_path}...")
    try:
        with open(answers_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("Answers saved.")
    except IOError as e:
        print(f"Error saving answers to {answers_path}: {e}")

    print("Processing complete.")

if __name__ == "__main__":
    # It's good practice to handle potential exceptions in main() as well,
    # or ensure all functions within main handle their specific exceptions.
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred in main execution: {e}")







