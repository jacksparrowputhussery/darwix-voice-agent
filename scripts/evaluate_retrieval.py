import os
import sys
import csv
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_base.retrieval.vector_store import VectorStore

def evaluate():
    queries = [
        "What is the minimum revenue required for a business loan?",
        "Do you provide funding to adult entertainment businesses?",
        "What are the available repayment terms?",
        "Can I apply if my credit score is 600?",
        "How long does the business need to be operating?"
    ]
    
    expected_sources = [
        "business_loan_policy.txt",
        "business_loan_policy.txt",
        "business_loan_policy.txt",
        "faq.csv",
        "business_loan_policy.txt"
    ]
    
    store = VectorStore()
    
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "retrieval_tests.csv"
    
    print("Evaluating Retrieval...")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "expected_source", "retrieved_source", "top_score", "relevance_reason", "verdict"])
        
        for q, expected in zip(queries, expected_sources):
            results = store.retrieve(q, top_k=3)
            
            if not results:
                writer.writerow([q, expected, "None", 0.0, "No results found", "incorrect"])
                continue
                
            top_result = results[0]
            retrieved_source = top_result.source
            
            # Simple heuristic evaluation for the prototype
            # Since we are using mock embeddings if API key is not set, the retrieval might not be semantically perfect.
            verdict = "incorrect"
            if expected in retrieved_source:
                verdict = "correct"
            
            writer.writerow([
                q, 
                expected, 
                retrieved_source, 
                round(top_result.score, 4), 
                top_result.content.replace('\n', ' ')[:100] + '...',
                verdict
            ])
            
    print(f"Evaluation complete. Results saved to {csv_path}")

if __name__ == "__main__":
    evaluate()
