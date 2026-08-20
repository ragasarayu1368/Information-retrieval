import os
import sys

# Ensure current dir is in path
sys.path.insert(0, os.path.abspath("."))

from database.database import init_db, SessionLocal
from database.models import User, Document, DocumentChunk, QueryLog, AuditLog
from auth.authentication import authenticate_user, verify_password
from services.user_service import UserService
from services.document_service import DocumentService
from services.evaluation_service import EvaluationService
from services.analytics_service import AnalyticsService
from rag.embeddings import get_embedding_generator
from rag.faiss_store import FaissVectorStore
from rag.inverted_index import InvertedIndex
from rag.rag_pipeline import run_rag_pipeline

def run_tests():
    print("==========================================================")
    print("STARTING END-TO-END RAG & RBAC VERIFICATION SUITE")
    print("==========================================================")

    # 1. Init DB
    print("[1/7] Initializing SQLite database and ORM tables...")
    init_db()
    db = SessionLocal()
    print("  [OK] Database initialized successfully.")

    # 2. Seed Users
    print("[2/7] Seeding and verifying demo user accounts...")
    UserService.seed_demo_users(db)
    
    emp_user = authenticate_user(db, "employee@example.com", "employee123")
    mgr_user = authenticate_user(db, "manager@example.com", "manager123")
    adm_user = authenticate_user(db, "admin@example.com", "admin123")
    
    assert emp_user is not None, "Employee auth failed!"
    assert mgr_user is not None, "Manager auth failed!"
    assert adm_user is not None, "Admin auth failed!"
    print(f"  [OK] Authenticated: {emp_user.email} ({emp_user.role}), {mgr_user.email} ({mgr_user.role}), {adm_user.email} ({adm_user.role})")

    # 3. Embedding & Vector Stores
    print("[3/7] Initializing FAISS ANN, Inverted Index, and Embedding models...")
    faiss_store = FaissVectorStore()
    inverted_index = InvertedIndex()
    embedding_gen = get_embedding_generator()
    print("  [OK] Stores and model initialized.")

    # 4. Seed Sample Enterprise Documents
    print("[4/7] Ingesting and indexing realistic enterprise documents...")
    DocumentService.seed_sample_documents(db, adm_user, faiss_store, inverted_index, embedding_gen)
    docs = DocumentService.list_documents(db)
    chunks = db.query(DocumentChunk).all()
    print(f"  [OK] Total Documents in DB: {len(docs)}, Total Chunks: {len(chunks)}, FAISS Vectors: {faiss_store.get_vector_count()}")
    assert len(docs) >= 6, "Expected at least 6 sample documents."

    # 5. Test RAG Pipeline Queries
    print("[5/7] Executing live RAG pipeline queries...")

    # Query 1: Work from home policy
    res1 = run_rag_pipeline(
        query="What is the work-from-home policy?",
        user=emp_user,
        db=db,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    print(f"\n  [Query 1] 'What is the work-from-home policy?'")
    print(f"  - Answer: {res1.answer}")
    print(f"  - Grounding Score: {res1.grounding_score*100:.1f}% (Is Grounded: {res1.is_grounded})")
    print(f"  - Top Source: {res1.retrieved_chunks[0].document_name if res1.retrieved_chunks else 'None'}")
    print(f"  - Timings: Total={res1.timings['total']*1000:.1f}ms, FAISS={res1.timings['faiss']*1000:.1f}ms, BM25={res1.timings['keyword']*1000:.1f}ms")
    assert res1.is_grounded, "Expected Query 1 to be grounded!"
    assert "remote" in res1.retrieved_chunks[0].document_name.lower(), "Expected Remote Work Policy!"

    # Query 2: Password requirements
    res2 = run_rag_pipeline(
        query="What are the password requirements?",
        user=emp_user,
        db=db,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    print(f"\n  [Query 2] 'What are the password requirements?'")
    print(f"  - Answer: {res2.answer}")
    print(f"  - Grounding Score: {res2.grounding_score*100:.1f}%")
    print(f"  - Top Source: {res2.retrieved_chunks[0].document_name if res2.retrieved_chunks else 'None'}")
    assert res2.is_grounded, "Expected Query 2 to be grounded!"

    # Query 3: Anti-Hallucination Guard Test (Stock Price)
    res3 = run_rag_pipeline(
        query="What is the company's stock price today?",
        user=emp_user,
        db=db,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    print(f"\n  [Query 3 - Anti-Hallucination Guard] 'What is the company's stock price today?'")
    print(f"  - Answer: {res3.answer}")
    print(f"  - Grounding Score: {res3.grounding_score*100:.1f}%")
    print(f"  - Blocked by Guard: {res3.blocked_by_guard}")
    assert not res3.is_grounded or res3.blocked_by_guard, "Expected Anti-Hallucination Guard to trigger!"
    print("  [OK] Anti-Hallucination Guard successfully blocked ungrounded query!")

    # 6. Test Security / Permission Filtering
    print("\n[6/7] Testing Permission-Aware Retrieval Barriers...")
    res_perm_emp = run_rag_pipeline(
        query="What is the confidential executive Q3 gross margin and revenue?",
        user=emp_user,  # Employee in Engineering
        db=db,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    # Employee should NOT retrieve the confidential finance report
    retrieved_doc_names = [c.document_name for c in res_perm_emp.retrieved_chunks]
    print(f"  - Employee Retrieved Chunks: {retrieved_doc_names}")
    assert not any("Financial" in d for d in retrieved_doc_names), "Security breach: Employee accessed confidential financial report!"
    print("  [OK] Confirmed: Employee successfully blocked from retrieving confidential management documents.")

    res_perm_adm = run_rag_pipeline(
        query="What is the confidential executive Q3 gross margin and revenue?",
        user=adm_user,  # Admin
        db=db,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    retrieved_adm_docs = [c.document_name for c in res_perm_adm.retrieved_chunks]
    print(f"  - Admin Retrieved Chunks: {retrieved_adm_docs}")
    assert any("Financial" in d for d in retrieved_adm_docs), "Admin should have access to financial report!"
    print("  [OK] Confirmed: Admin has global access across all departmental tiers.")

    # 7. Run Benchmark Evaluation
    print("\n[7/7] Executing automated benchmark evaluation suite...")
    eval_res = EvaluationService.run_benchmark_suite(
        db=db,
        admin_user=adm_user,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    print(f"  - Retrieval Accuracy: {eval_res['retrieval_accuracy']}%")
    print(f"  - Source Accuracy: {eval_res['source_accuracy']}%")
    print(f"  - Grounded Rate: {eval_res['grounded_rate']}%")
    print(f"  - Avg Latency: {eval_res['avg_latency_ms']:.2f} ms")
    assert eval_res['retrieval_accuracy'] >= 80.0, f"Expected high retrieval accuracy, got {eval_res['retrieval_accuracy']}%"

    db.close()
    print("\n==========================================================")
    print("ALL VERIFICATION TESTS PASSED PERFECTLY (100% SUCCESS)!")
    print("==========================================================")

if __name__ == "__main__":
    run_tests()
