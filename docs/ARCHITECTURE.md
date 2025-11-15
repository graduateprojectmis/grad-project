## 類別設計

### 核心類別 UML

```
┌─────────────────────────────────┐
│      BaseEmbeddingService       │
│─────────────────────────────────│
│ + generate_embedding()          │
└─────────────────────────────────┘
         ▲                ▲
         │                │
         │                │
┌────────┴────────┐  ┌───┴──────────────┐
│ OpenAIEmbedding │  │ GeminiEmbedding  │
│    Service      │  │    Service       │
│─────────────────│  │──────────────────│
│ - api_key       │  │ - api_key        │
│ - model         │  │ - model          │
│─────────────────│  │──────────────────│
│ + __init__()    │  │ + __init__()     │
│ + generate_     │  │ + generate_      │
│   embedding()   │  │   embedding()    │
└─────────────────┘  └──────────────────┘
```

```
┌─────────────────────────────────┐
│       DatabaseService           │
│─────────────────────────────────│
│ - client: PersistentClient      │
│ - collection: Collection        │
│ - db_path: str                  │
│─────────────────────────────────│
│ + __init__(db_path, collection) │
│ + insert(ids, docs, embeddings) │
│ + query(embedding, n_results)   │
│ + count()                       │
│ + clear()                       │
│ + close()                       │
└─────────────────────────────────┘
```

```
┌─────────────────────────────────┐
│         LLMService              │
│─────────────────────────────────│
│ - api_key: str                  │
│ - model: str                    │
│ - temperature: float            │
│─────────────────────────────────│
│ + __init__(api_key, model)      │
│ + generate_answer(q, context)   │
│ + generate_summary(text)        │
│ - _build_prompt(q, context)     │
└─────────────────────────────────┘
```

## 錯誤處理流程

```
┌──────────────┐
│ API 請求     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 驗證輸入     │──► 驗證失敗 ──► ValidationError
└──────┬───────┘
       │ 驗證通過
       ▼
┌──────────────┐
│ 執行業務邏輯  │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ 可能的錯誤           │
├──────────────────────┤
│ • APIKeyError        │──► HTTP 400
│ • DatabaseError      │──► HTTP 503
│ • EmbeddingError     │──► HTTP 500
│ • AppException       │──► HTTP 400
│ • Other Exception    │──► HTTP 500
└──────────────────────┘
       │
       ▼
┌──────────────┐
│ 全域例外處理  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 記錄日誌     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 返回錯誤回應  │
│ (JSON)       │
└──────────────┘
```
