import json
import time

from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex, SentenceTransformerEmbedder
from app.retrieval.service import search
from app.generation.llm_client import LLMClient
from app.generation.drafting import draft_answer
from app.decomposition.service import decompose
from app.verification.nli import load_nli_model
from app.verification.service import verify

USE_NLI = True  # False = LLM fallback, low confidence NLI model

def main():
    # Loads prepared corpus
    with open("data/corpus.json", "r", encoding="utf-8") as file:
        documents = json.load(file)

    if not documents:
        print("The corpus is empty. Build the corpus first.")
        return

    print(f"Loaded {len(documents)} document chunks.")

    # Builds retrieval indexes
    bm25_index = BM25Index(documents)
    embedder = SentenceTransformerEmbedder()
    dense_index = DenseIndex(documents, embedder)

    print("BM25 and dense indexes are ready.")

    # User asks research question
    question = input("\nEnter a research question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # Retrieves + combines relevant passages
    passages = search(
        bm25_index,
        dense_index,
        question,
        k=5,
    )

    print("\n--- RETRIEVED PASSAGES ---")

    for i, passage in enumerate(passages, start=1):
        print(f"\nPassage {i}")
        print(f"ID: {passage['id']}")
        print(f"Score: {passage['score']:.4f}")
        print(f"Text: {passage['text'][:500]}")

    # Generates answer using retrieved passages
    llm = LLMClient()

    result = draft_answer(
        llm,
        question,
        passages,
    )

    print("\n--- GENERATED ANSWER ---")
    print(result["answer"] or "No sufficient answer generated.")
    print(f"Sufficient: {result['sufficient']}")

    # Splits answer into atomic claims
    claims = decompose(llm, result["answer"])

    print("\n--- DECOMPOSED CLAIMS ---")

    if not claims:
        print("No claims to display.")
        return

    for i, claim in enumerate(claims, start=1):
        print(f"{i}. {claim}")

    # Verifies each claim against its own retrieved evidence
    nli = None
    if USE_NLI:
        print("\nLoading NLI model...")
        nli = load_nli_model()

    print("\n--- VERIFICATION ---")

    search_seconds = 0.0
    verify_seconds = {}
    verify_counts = {}

    for i, claim in enumerate(claims, start=1):
        started = time.time()
        evidence = search(bm25_index, dense_index, claim, k=3)
        search_seconds += time.time() - started

        print(f"\nClaim {i}: {claim}")

        for passage in evidence:
            started = time.time()
            verdict = verify(nli, llm, claim, passage["text"])
            elapsed = time.time() - started

            method = verdict["method"]
            verify_seconds[method] = verify_seconds.get(method, 0.0) + elapsed
            verify_counts[method] = verify_counts.get(method, 0) + 1

            print(f"  {verdict['status']} (decided by {method})")
            if verdict["scores"]:
                print("    Scores: " + ", ".join(f"{k} {v:.2f}" for k, v in verdict["scores"].items()))
            print(f"    ID: {passage['id']}")
            print(f"    Text: {passage['text'][:150]}")

    print("\n--- TIMING ---")
    print(f"Evidence search: {search_seconds:.1f}s")
    for method, seconds in verify_seconds.items():
        count = verify_counts[method]
        print(f"verify, decided by {method}: {count} pairs, {seconds:.1f}s total, {seconds / count:.2f}s each")

if __name__ == "__main__":
    main()