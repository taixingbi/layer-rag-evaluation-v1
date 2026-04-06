

###
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

### Run

```bash
python3 main.py -i dataset/dataset-gold-test-1.0.0.json -o result/dataset-gold-test-1.0.0.json --max-concurrency 5
```

`--max-concurrency` limits how many `/v1/rag/query` requests are **in flight** from this client; progress lines print as each slot starts work. Total time still depends on the RAG server (it may queue requests). Use **Ctrl+C** to stop; **Ctrl+Z** suspends the job (`zsh: suspended`).

