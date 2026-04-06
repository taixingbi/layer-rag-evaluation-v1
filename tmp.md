


### reference api 
##### input
curl -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "k": 5,
    "k_max": 40
  }'

##### output
{
  "question": "what is taixing visa",
  "answer": "Taixing Bi's visa status is H4 EAD [1].",
  "citations": [
    {
      "id": "c1",
      "rank": 1,
      "chunk_id": "16310064034107487855",
      "source": "profile.json",
      "rrf_score": 0.0325,
      "excerpt": "id: profile_005\nquestion: What is the candidate's visa status?\nanswer: H4 EAD\ncategory: qa\ntags: taixing-bi\nvisa\nwork-authorization\nh4-ead\nsource: profile.json\ndate: 2026-04-04\nauthor: Taixing Bi"
    }
  ],
  "meta": {
    "session_id": "ses-xyz789",
    "request_id": "req-abc123",
    "latency_ms": 5191,
    "model": "Qwen/Qwen2.5-7B-Instruct"
  }
}

### file api 
##### input
[
  {
    "input": "Who is Taixing?",
    "inference-output": "",
    "output": "Taixing is an AI Infrastructure Engineer with over 7 years of experience building production-grade LLM systems, RAG pipelines, and distributed machine learning platforms."
  },
  {
    "input": "What is Taixing's current role?",
    "inference-output": "",
    "output": "AI Infrastructure Engineer"
  }
]

##### output
[
    {
        "question": "what is taixing visa",
        "answer": "Taixing Bi's visa status is H4 EAD [1].",
        "citations": [
            {
            "id": "c1",
            "rank": 1,
            "chunk_id": "16310064034107487855",
            "source": "profile.json",
            "rrf_score": 0.0325,
            "excerpt": "id: profile_005\nquestion: What is the candidate's visa status?\nanswer: H4 EAD\ncategory: qa\ntags: taixing-bi\nvisa\nwork-authorization\nh4-ead\nsource: profile.json\ndate: 2026-04-04\nauthor: Taixing Bi"
            }
        ],
        "meta": {
            "session_id": "ses-xyz789",
            "request_id": "req-abc123",
            "latency_ms": 5191,
            "model": "Qwen/Qwen2.5-7B-Instruct"
        },
        "metrics": {
            "faithfulness_grounding": 0.9125,
            "answer_correctness": 0.6932,
            "context_relevance": null,
            "answer_relevance": 0.7609,
            "hallucination_rate_proxy": 0.3068
        }
    }
]




python3 rag_query.py -i dataset/dataset-gold-test-1.0.0.json -o result/dataset-gold-test-1.0.0.json
