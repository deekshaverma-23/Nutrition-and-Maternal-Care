import sys
import os
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset

# ---------------- PATH SETUP ----------------

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# ---------------- RAGAS IMPORTS ----------------

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# ---------------- LANGCHAIN / OLLAMA ----------------

from langchain_ollama import ChatOllama, OllamaLLM, OllamaEmbeddings

# ---------------- PROJECT IMPORTS ----------------

from core.rag import get_rag_chain
from core.llm import get_llms


# ---------------- CONFIG ----------------

DATA_PATH = Path("evaluation/sample_questions.json")
OUT_PATH = Path("evaluation/rag_results.csv")


# ---------------- RAG RUNNER ----------------


def run_rag_query(rag_chain, question: str):

    result = rag_chain.invoke({"input": question})

    docs = result.get("context") or result.get("documents") or []

    contexts = [d.page_content for d in docs]

    return result["answer"], contexts


# ---------------- MAIN EVAL ----------------


def run_eval():

    print("\n--- Initializing models ---")

    llms = get_llms()
    rag_chain = get_rag_chain(llms)

    # Judge LLM (deterministic)
    judge_chat = ChatOllama(model="llama3")
    judge_llm = LangchainLLMWrapper(judge_chat)

    # Generation LLM (same as app)
    gen_llm = llms["llm"]

    # Embeddings wrapper for RAGAS
    base_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    ragas_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

    print("--- Loading eval dataset ---")

    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    data_samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print("--- Step 1: Generating answers from RAG ---")

    for row in raw:

        q = row["question"]

        ans, ctx = run_rag_query(rag_chain, q)

        data_samples["question"].append(q)
        data_samples["answer"].append(ans)
        data_samples["contexts"].append(ctx)

        # For now we treat generated answer as GT placeholder
        data_samples["ground_truth"].append(ans)

        print("✔", q[:70])

    dataset = Dataset.from_dict(data_samples)

    print("\n--- Step 2: Running LOCAL RAGAS evaluation ---")

    metrics = [
        Faithfulness(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
        AnswerRelevancy(
            llm=judge_llm,
            embeddings=ragas_embeddings,
        ),
    ]

    run_config = RunConfig(
        max_workers=1,
        timeout=300,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
    )

    df = result.to_pandas()

    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("\n" + "=" * 45)
    print("        LOCAL RAG EVALUATION DONE")
    print("=" * 45)
    print(df.mean(numeric_only=True))
    print("\nSaved to:", OUT_PATH)


if __name__ == "__main__":
    run_eval()
