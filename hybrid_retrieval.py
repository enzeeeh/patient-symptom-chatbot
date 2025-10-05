"""
Hybrid Retrieval System for Medical Chatbot
Combines:
1. LangChain RAG for local medical guidelines
2. Exa API for real-time medical research
3. Intelligent source ranking and combination
"""

import os
import glob
from typing import List, Dict, Optional
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Try to import Exa, make it optional
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    st.warning("Exa not available. Install with: pip install exa-py")

class HybridMedicalRetriever:
    def __init__(self, gemini_api_key: str, exa_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key
        self.exa_api_key = exa_api_key
        self.local_vectorstore = None
        self.exa_client = None
        
        # Initialize components
        self._setup_local_rag()
        if exa_api_key and EXA_AVAILABLE:
            self._setup_exa()
    
    def _setup_local_rag(self):
        """Setup LangChain RAG for local medical guidelines"""
        try:
            # Use Google's embeddings for better medical understanding
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.gemini_api_key
            )
            
            # Load and process medical guidelines
            documents = self._load_medical_guidelines()
            if documents:
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                splits = text_splitter.split_documents(documents)
                
                # Create vector store
                self.local_vectorstore = FAISS.from_documents(splits, embeddings)
                st.success(f"✅ Loaded {len(documents)} medical guidelines into RAG system")
            else:
                st.warning("⚠️ No medical guidelines found")
                
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                st.warning("⚠️ Gemini embeddings quota exceeded. Using fallback embeddings.")
            else:
                st.error(f"❌ Error setting up Gemini RAG: {e}")
            
            # Fallback to HuggingFace embeddings
            try:
                st.info("🔄 Switching to HuggingFace embeddings (free, no quota limits)...")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                documents = self._load_medical_guidelines()
                if documents:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(documents)
                    self.local_vectorstore = FAISS.from_documents(splits, embeddings)
                    st.success(f"✅ Loaded {len(documents)} guidelines (HuggingFace embeddings)")
                else:
                    st.warning("⚠️ No medical guidelines found to load")
            except Exception as fallback_error:
                st.error(f"❌ Fallback embeddings also failed: {fallback_error}")
                st.info("💡 Hybrid mode will use basic search fallback")
    
    def _setup_exa(self):
        """Setup Exa API for real-time web search"""
        try:
            self.exa_client = Exa(api_key=self.exa_api_key)
            st.success("✅ Exa API connected for real-time medical research")
        except Exception as e:
            st.warning(f"⚠️ Exa setup failed: {e}")
    
    def _load_medical_guidelines(self):
        """Load medical guidelines from the guidelines folder"""
        from langchain.schema import Document
        
        guideline_folder = os.path.join(os.path.dirname(__file__), "guidelines")
        documents = []
        
        # Load markdown files
        md_files = glob.glob(os.path.join(guideline_folder, "*.md"))
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            'source': os.path.basename(file_path),
                            'type': 'local_guideline'
                        }
                    ))
            except Exception as e:
                st.warning(f"Could not load {file_path}: {e}")
        
        return documents
    
    def search_local_guidelines(self, query: str, k: int = 3) -> List[Dict]:
        """Search local medical guidelines using RAG"""
        if not self.local_vectorstore:
            # Fallback to simple keyword search
            return self._fallback_keyword_search(query, k)
        
        try:
            docs = self.local_vectorstore.similarity_search(query, k=k)
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'type': 'local_guideline',
                    'relevance_score': 0.8  # High relevance for local guidelines
                })
            return results
        except Exception as e:
            st.warning(f"RAG search failed, using keyword fallback: {e}")
            return self._fallback_keyword_search(query, k)
    
    def _fallback_keyword_search(self, query: str, k: int = 3) -> List[Dict]:
        """Fallback keyword search when RAG is not available"""
        documents = self._load_medical_guidelines()
        if not documents:
            return []
        
        query_lower = query.lower()
        scored_docs = []
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            # Simple keyword scoring
            score = sum(1 for word in query_lower.split() if word in content_lower)
            if score > 0:
                scored_docs.append({
                    'content': doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'type': 'local_guideline',
                    'relevance_score': 0.6,  # Lower score for keyword search
                    'keyword_score': score
                })
        
        # Sort by keyword score
        scored_docs.sort(key=lambda x: x.get('keyword_score', 0), reverse=True)
        return scored_docs[:k]
    
    def search_web_research(self, query: str, num_results: int = 2) -> List[Dict]:
        """Search web for latest medical research using Exa"""
        if not self.exa_client:
            return []
        
        try:
            # Focus on medical and academic sources
            medical_query = f"medical research {query} symptoms treatment diagnosis"
            
            response = self.exa_client.search_and_contents(
                query=medical_query,
                num_results=num_results,
                include_domains=["pubmed.ncbi.nlm.nih.gov", "who.int", "cdc.gov", "mayoclinic.org", "medlineplus.gov"],
                text=True
            )
            
            results = []
            for item in response.results:
                results.append({
                    'content': item.text[:1000] + "..." if len(item.text) > 1000 else item.text,
                    'source': item.url,
                    'title': item.title,
                    'type': 'web_research',
                    'relevance_score': item.score if hasattr(item, 'score') else 0.6
                })
            
            return results
            
        except Exception as e:
            st.warning(f"Web search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, symptoms: List[str], conditions: List[str]) -> Dict:
        """
        Perform hybrid search combining local guidelines and web research
        """
        # Enhanced query with medical context
        enhanced_query = f"{query}. Symptoms: {', '.join(symptoms)}. Conditions: {', '.join([c.get('name', '') for c in conditions])}"
        
        # Search local guidelines (primary source)
        local_results = self.search_local_guidelines(enhanced_query, k=3)
        
        # Search web for supplementary information (if available)
        web_results = []
        if self.exa_client:
            web_results = self.search_web_research(enhanced_query, num_results=2)
        
        # Combine and rank results
        all_results = local_results + web_results
        
        # Sort by relevance score (local guidelines get priority)
        # Handle None values in relevance_score
        all_results.sort(key=lambda x: x.get('relevance_score', 0) or 0, reverse=True)
        
        return {
            'local_results': local_results,
            'web_results': web_results,
            'combined_results': all_results[:5],  # Top 5 results
            'total_sources': len(all_results)
        }
    
    def get_context_for_gemini(self, search_results: Dict) -> str:
        """
        Prepare context from hybrid search results for Gemini
        """
        context_parts = []
        
        # Add local guidelines context
        if search_results['local_results']:
            context_parts.append("=== MEDICAL GUIDELINES (Primary Sources) ===")
            for result in search_results['local_results']:
                context_parts.append(f"Source: {result['source']}")
                context_parts.append(f"Content: {result['content'][:500]}...")
                context_parts.append("")
        
        # Add web research context
        if search_results['web_results']:
            context_parts.append("=== LATEST MEDICAL RESEARCH (Supplementary) ===")
            for result in search_results['web_results']:
                context_parts.append(f"Source: {result.get('title', 'Web Source')}")
                context_parts.append(f"URL: {result['source']}")
                context_parts.append(f"Content: {result['content'][:300]}...")
                context_parts.append("")
        
        return "\n".join(context_parts)

# Cached instance
@st.cache_resource
def get_hybrid_retriever(gemini_api_key: str, exa_api_key: Optional[str] = None):
    """Get cached instance of hybrid retriever"""
    return HybridMedicalRetriever(gemini_api_key, exa_api_key)