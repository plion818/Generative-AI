# Knowledge QA System

本專案為一個自動化知識問答系統，結合 PDF 文字抽取、語意向量檢索（FAISS）、與大型語言模型（LLaMA），可根據 PDF 文件內容自動回答多個問題。

## 需求安裝

請先安裝 Python 3.8 以上，並於專案目錄下執行：

```powershell
pip install -r requirements.txt
```

## 使用說明

1. **準備資料**
   - 將知識庫 PDF 檔案命名為 `book.pdf`，放在專案根目錄。
   - 準備問題檔 `queries.json`，格式如下：

```json
[
  {"id": 1, "question": "請問 LLaMA 是什麼？"},
  {"id": 2, "question": "本書的主題是什麼？"}
]
```

2. **執行腳本**

```powershell
python knowledge_qa.py
```

3. **輸出結果**
   - 程式會自動產生 `answers.json`，格式如下：

```json
[
  {
    "query_id": 1,
    "question": "請問 LLaMA 是什麼？",
    "answer": "LLaMA 是一種大型語言模型，由 Meta AI 開發。"
  },
  {
    "query_id": 2,
    "question": "本書的主題是什麼？",
    "answer": "文件中未找到相關資訊"
  }
]
```

## 注意事項

- 預設使用 `BAAI/bge-small-zh-v1.5` 向量模型與 `openlm-research/open_llama_7b`，首次執行會自動下載模型，需有網路連線。
- 若硬體資源有限，可考慮更換較小的 LLaMA 模型。
- 若 `book.pdf` 或 `queries.json` 檔案不存在，程式會自動結束並提示錯誤。

## 檔案說明

- `knowledge_qa.py`：主程式，執行知識問答流程。
- `requirements.txt`：Python 依賴套件列表。
- `book.pdf`：知識庫 PDF 文件。
- `queries.json`：問題集（JSON 格式）。
- `answers.json`：自動產生的問答結果。

---

如有問題請於 Issues 回報。
