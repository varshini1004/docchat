
import csv
import time
from query import load_vectorstore, build_context, PROMPT_TEMPLATE, LLM_MODEL, TOP_K
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ---- EDIT THIS: add 5-10 questions relevant to YOUR document ----
TEST_QUESTIONS = [
    "What is supervised learning?",
    "What does the document say about boolean variables?",
    "What is data science?",
    "How does file access work?",
    "What is the purpose of matplotlib?",
]
# -------------------------------------------------------------------

OUTPUT_CSV = "eval_results.csv"


def run_evaluation():
    vectorstore = load_vectorstore()
    llm = ChatGroq(model=LLM_MODEL, temperature=0)

    rows = []
    for question in TEST_QUESTIONS:
        start = time.time()

        retrieved_docs = vectorstore.similarity_search(question, k=TOP_K)
        context, sources = build_context(retrieved_docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = llm.invoke(prompt)

        elapsed = round(time.time() - start, 2)
        num_sources = len(retrieved_docs)

        # Simple heuristic flags -- not perfect, but gives you something
        # concrete to report and iterate on.
        answer_lower = response.content.lower()
        said_dont_know = any(
            phrase in answer_lower
            for phrase in ["don't know", "do not know", "no information", "not mentioned"]
        )

        rows.append({
            "question": question,
            "answer": response.content,
            "num_sources_retrieved": num_sources,
            "sources": sources.replace("\n", " | "),
            "response_time_sec": elapsed,
            "answer_says_dont_know": said_dont_know,
        })

        print(f"[{len(rows)}/{len(TEST_QUESTIONS)}] {question}  ({elapsed}s)")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Print a quick summary
    avg_time = sum(r["response_time_sec"] for r in rows) / len(rows)
    dont_know_count = sum(r["answer_says_dont_know"] for r in rows)

    print("\n--- Evaluation Summary ---")
    print(f"Total questions tested: {len(rows)}")
    print(f"Average response time: {avg_time:.2f}s")
    print(f"Questions the model couldn't answer from context: {dont_know_count}/{len(rows)}")
    print(f"Full results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_evaluation()
