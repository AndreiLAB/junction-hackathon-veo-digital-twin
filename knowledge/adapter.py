import sqlite3
from typing import List
from models import DocumentMatch
from scripts.kb_search import search

DEVICE_CLASSES = {
    "abb_relion_615": ("ABB 615 protection relay", "ABB 615"),
    "relay_front": ("ABB 615 protection relay", "ABB 615"),
    "relay_rear": ("ABB 615 protection relay", "ABB 615"),
}

def find_documents(class_name: str, db_path: str = "data/knowledge.db") -> List[DocumentMatch]:
    """
    Looks up manuals for a given class_name (e.g. abb_relion_615).
    """
    if class_name not in DEVICE_CLASSES:
        return []
        
    _, model_name = DEVICE_CLASSES[class_name]
    
    try:
        con = sqlite3.connect(db_path)
    except Exception:
        return []
        
    # Query something generic since we just want the document
    hits = search(con, q=model_name, k=3, model=model_name)
    
    matches = []
    for h in hits:
        matches.append(DocumentMatch(
            document_id=h["document_id"],
            title=h["document"],
            section_path=h["section_path"],
            page_start=h["page_start"],
            text_snippet=h["text"][:200]
        ))
        
    con.close()
    return matches
