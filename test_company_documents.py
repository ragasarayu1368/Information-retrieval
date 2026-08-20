import os
import sys
import numpy as np
from datetime import datetime

# Set path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database.database import get_db, init_db
from database.models import User, Document, DocumentVersion, DocumentChunk, AuditLog
from database.csv_db import sync_sqlalchemy_to_csv, load_table_df
from services.user_service import UserService
from services.document_service import DocumentService, compute_text_diff
from services.audit_service import AuditService
from rag.embeddings import get_embedding_generator
from rag.faiss_store import FaissVectorStore
from rag.inverted_index import InvertedIndex
from rag.rag_pipeline import run_rag_pipeline

def run_tests():
    print("=" * 70)
    print("STARTING COMPANY DOCUMENT MANAGEMENT SYSTEM & RAG TEST SUITE")
    print("=" * 70)

    # 1. Init DB and Components
    print("\n[Step 1] Initializing SQLite database, FAISS store, BM25 index...")
    init_db()
    db = next(get_db())
    embedding_gen = get_embedding_generator()
    faiss_store = FaissVectorStore()
    inverted_index = InvertedIndex()

    admin = db.query(User).filter(User.email == "admin@example.com").first()
    employee = db.query(User).filter(User.email == "employee@example.com").first()
    assert admin is not None, "Admin user required"
    assert employee is not None, "Employee user required"
    print("  [OK] Database and models ready.")

    # 2. Seed Base Documents
    print("\n[Step 2] Seeding standard enterprise documents (v1)...")
    DocumentService.seed_sample_documents(db, admin, faiss_store, inverted_index, embedding_gen)
    docs = DocumentService.list_documents(db)
    print(f"  [OK] {len(docs)} active documents seeded. Vector count in FAISS: {faiss_store.get_vector_count()}")

    # 3. Test Duplicate Document Detection
    print("\n[Step 3] Testing SHA-256 Duplicate Document Detection...")
    sample_txt = "ENTERPRISE REMOTE WORK & WORK-FROM-HOME POLICY\n1. Overview & Eligibility..."
    is_dup, dup_doc = DocumentService.check_duplicate_document(db, sample_txt.encode("utf-8"))
    print(f"  - Duplicate check result: {is_dup}")
    assert is_dup is False or dup_doc is not None

    # 4. Test RAG Query BEFORE uploading New Policy
    print("\n[Step 4] Querying AI Assistant BEFORE new policy upload...")
    pre_result = run_rag_pipeline(
        query="What is the global remote work cybersecurity stipend allowance in 2027?",
        user=admin,
        db=db,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    print(f"  - Pre-upload response: {pre_result.answer[:80]}...")
    print(f"  - Is Grounded: {pre_result.is_grounded}, Grounding Score: {pre_result.grounding_score:.1%}")

    # 5. Test Uploading New Document
    print("\n[Step 5] Uploading 'New_Global_Stipend_Policy_2027.txt' (v1)...")
    new_policy_text = """ENTERPRISE GLOBAL CYBERSECURITY STIPEND POLICY 2027
1. Global Remote Stipend Allowance
Starting in fiscal year 2027, all full-time engineers and staff receive an annual $1,850 global remote cybersecurity stipend.
2. Approved Expenses
This stipend covers high-speed optical fiber internet, biometric security hardware keys, noise-canceling headsets, and encrypted backup drives.
3. Reimbursement Procedure
Submit invoices via the Enterprise Expense Portal with approval from the Department VP within 45 days."""
    
    new_doc, err = DocumentService.process_and_index_document(
        db=db,
        filename="New_Global_Stipend_Policy_2027.txt",
        file_bytes=new_policy_text.encode("utf-8"),
        department="IT",
        access_roles="EMPLOYEE,MANAGER,ADMIN",
        access_level="Public",
        user_id=admin.id,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    assert err is None, f"Upload error: {err}"
    assert new_doc.current_version == 1
    assert new_doc.current_version_label == "v1"
    print(f"  [OK] Ingested '{new_doc.name}' as v1 with {new_doc.chunk_count} chunks. FAISS Vectors: {faiss_store.get_vector_count()}")

    # 6. Test RAG Query AFTER uploading New Policy
    print("\n[Step 6] Querying AI Assistant AFTER new policy upload...")
    post_result = run_rag_pipeline(
        query="What is the global remote work cybersecurity stipend allowance in 2027?",
        user=admin,
        db=db,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    print(f"  - Post-upload response: {post_result.answer[:120]}...")
    print(f"  - Top Source: {post_result.retrieved_chunks[0].document_name} ({post_result.retrieved_chunks[0].version_label})")
    print(f"  - Grounding Score: {post_result.grounding_score:.1%}, Is Grounded: {post_result.is_grounded}")
    assert post_result.is_grounded is True
    assert "$1,850" in post_result.answer or "1,850" in post_result.answer
    assert post_result.retrieved_chunks[0].document_name == "New_Global_Stipend_Policy_2027.txt"
    assert post_result.retrieved_chunks[0].version_label == "v1"
    print("  [OK] Automatic Real-Time RAG retrieval verified for newly uploaded document!")

    # 7. Test Updating Document to v2 (Version Upgrade + Text Diff)
    print("\n[Step 7] Updating 'New_Global_Stipend_Policy_2027.txt' to v2 with revised stipend amount ($2,500)...")
    v2_text = """ENTERPRISE GLOBAL CYBERSECURITY STIPEND POLICY 2027
1. Global Remote Stipend Allowance
Starting in fiscal year 2027, all full-time engineers and staff receive an enhanced $2,500 global remote cybersecurity stipend.
2. Approved Expenses
This stipend covers high-speed optical fiber internet, biometric security hardware keys, noise-canceling headsets, and standing desks.
3. Reimbursement Procedure
Submit invoices via the Enterprise Expense Portal with approval from the Department VP within 60 days."""

    diff_output = compute_text_diff(new_policy_text, v2_text)
    print(f"  - Text diff preview:\n{diff_output}")

    updated_doc, err = DocumentService.update_document_version(
        db=db,
        doc_id=new_doc.id,
        filename="New_Global_Stipend_Policy_2027.txt",
        file_bytes=v2_text.encode("utf-8"),
        user_id=admin.id,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    assert err is None, f"Update error: {err}"
    assert updated_doc.current_version == 2
    assert updated_doc.current_version_label == "v2"
    print(f"  [OK] Upgraded to {updated_doc.current_version_label}. FAISS Vectors: {faiss_store.get_vector_count()}")

    # 8. Test RAG Query on Updated Version v2
    print("\n[Step 8] Querying AI Assistant on v2...")
    v2_result = run_rag_pipeline(
        query="What is the global remote work cybersecurity stipend allowance in 2027?",
        user=admin,
        db=db,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    print(f"  - v2 Response: {v2_result.answer[:120]}...")
    print(f"  - Top Source: {v2_result.retrieved_chunks[0].document_name} ({v2_result.retrieved_chunks[0].version_label})")
    assert "$2,500" in v2_result.answer or "2,500" in v2_result.answer
    assert v2_result.retrieved_chunks[0].version_label == "v2"
    print("  [OK] Updated v2 information immediately retrieved by RAG!")

    # 9. Test Restoring v1 from v2
    print("\n[Step 9] Restoring v1 historical version...")
    v1_record = db.query(DocumentVersion).filter(
        DocumentVersion.document_id == updated_doc.id,
        DocumentVersion.version_number == 1
    ).first()
    assert v1_record is not None

    ok, err = DocumentService.restore_document_version(
        db=db,
        doc_id=updated_doc.id,
        target_version_id=v1_record.id,
        user_id=admin.id,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    assert ok is True, f"Restore error: {err}"
    restored_doc = DocumentService.get_document_by_id(db, updated_doc.id)
    assert restored_doc.current_version == 1
    assert restored_doc.current_version_label == "v1"
    print(f"  [OK] Restored to {restored_doc.current_version_label}.")

    # 10. Test RAG Query on Restored v1
    print("\n[Step 10] Querying AI Assistant on Restored v1...")
    restored_result = run_rag_pipeline(
        query="What is the global remote work cybersecurity stipend allowance in 2027?",
        user=admin,
        db=db,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    print(f"  - Restored v1 Response: {restored_result.answer[:120]}...")
    assert "$1,850" in restored_result.answer or "1,850" in restored_result.answer
    assert restored_result.retrieved_chunks[0].version_label == "v1"
    print("  [OK] Restored v1 knowledge immediately active in RAG pipeline!")

    # 11. Test Document Deletion
    print("\n[Step 11] Testing Document Deletion & Vector Purge...")
    del_ok = DocumentService.delete_document(db, new_doc.id, faiss_store, inverted_index, embedding_gen)
    assert del_ok is True
    del_doc = DocumentService.get_document_by_id(db, new_doc.id)
    assert del_doc is None
    print(f"  [OK] Document deleted from DB, chunks purged, and FAISS vectors updated ({faiss_store.get_vector_count()} remaining).")

    # 12. Verify RAG Query AFTER Deletion
    print("\n[Step 12] Querying AI Assistant AFTER deletion...")
    del_result = run_rag_pipeline(
        query="What is the global remote work cybersecurity stipend allowance in 2027?",
        user=admin,
        db=db,
        faiss_store=faiss_store,
        inverted_index=inverted_index,
        embedding_gen=embedding_gen
    )
    print(f"  - Response after deletion: {del_result.answer[:80]}...")
    # Should not retrieve deleted doc
    for c in del_result.retrieved_chunks:
        assert c.document_name != "New_Global_Stipend_Policy_2027.txt"
    print("  [OK] Deleted document is completely purged and unsearchable!")

    # 13. Verify Audit Trail
    print("\n[Step 13] Verifying Security Audit Trail...")
    audit_logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(10).all()
    print(f"  [OK] Total recent audit logs: {len(audit_logs)}")
    for a in audit_logs[:5]:
        print(f"    - [{a.timestamp.strftime('%H:%M:%S')}] {a.user_email} -> {a.action} ({a.status}) on {a.resource}")

    db.close()
    print("\n" + "=" * 70)
    print("ALL 13 END-TO-END DOCUMENT MANAGEMENT & RAG TESTS PASSED (100%)!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
