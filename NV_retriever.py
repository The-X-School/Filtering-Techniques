import os
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import warnings
import json
import gc
import re
from typing import List, Dict, Any, Tuple, Optional

warnings.filterwarnings("ignore")
os.environ["TORCH_LOGS"] = "0"

class FunctionCallingRetriever:
    """
    Specialized retriever for function calling content from the ClimbLab dataset.
    Filters, embeds, and retrieves function calling-relevant documents.
    """
    
    def __init__(self):
        """Initialize the function calling retriever"""
        self.function_calling_patterns = self._compile_function_calling_patterns()
        self.model = None
        self.embeddings = None
        self.texts = None
        self.function_scores = None
        
    def _compile_function_calling_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        Compile regex patterns specifically for detecting function calling content
        """
        patterns = [
            # Voice assistant commands
            (re.compile(r'\b(?:hey\s+(?:siri|alexa|google)|ok\s+google|alexa)', re.IGNORECASE), 
             "Voice assistant wake words", 10),
            
            # Function/method calls
            (re.compile(r'\w+\s*\([^)]*\)', re.IGNORECASE), 
             "Function calls with parentheses", 8),
            
            # API calls and endpoints
            (re.compile(r'\b(?:GET|POST|PUT|DELETE|PATCH)\s+[/\w\-\.]+', re.IGNORECASE), 
             "HTTP API calls", 9),
            
            # Command patterns
            (re.compile(r'\b(?:execute|run|call|invoke|trigger|activate)\s+\w+', re.IGNORECASE), 
             "Command execution patterns", 7),
            
            # Mobile device actions
            (re.compile(r'\b(?:turn\s+on|turn\s+off|set|open|close|start|stop)\s+(?:the\s+)?\w+', re.IGNORECASE), 
             "Device control commands", 8),
            
            # Programming function definitions
            (re.compile(r'\b(?:def|function|func|method)\s+\w+\s*\(', re.IGNORECASE), 
             "Function definitions", 9),
            
            # JSON-RPC and API patterns
            (re.compile(r'\{[^}]*"method"\s*:\s*"[^"]*"[^}]*\}', re.IGNORECASE), 
             "JSON-RPC method calls", 10),
            
            # Tool/library function calls
            (re.compile(r'\b(?:requests|fetch|axios|curl)\.\w+\(', re.IGNORECASE), 
             "HTTP library calls", 8),
            
            # Smart home commands
            (re.compile(r'\b(?:lights?|thermostat|door|window|alarm|camera)\b', re.IGNORECASE), 
             "Smart home devices", 6),
            
            # Programming keywords
            (re.compile(r'\b(?:import|from|require|include)\s+\w+', re.IGNORECASE), 
             "Module imports", 5),
        ]
        return patterns
    
    def calculate_function_calling_score(self, text: str) -> float:
        """
        Calculate how relevant a text is to function calling
        """
        if not text:
            return 0.0
        
        total_score = 0
        text_length = len(text)
        
        for pattern, description, weight in self.function_calling_patterns:
            matches = len(pattern.findall(text))
            if matches > 0:
                total_score += matches * weight
        
        # Normalize by text length to avoid bias toward longer texts
        normalized_score = total_score / (text_length / 1000) if text_length > 0 else 0
        return min(normalized_score, 100.0)  # Cap at 100
    
    def filter_function_calling_content(self, docs: List[str], min_score: float = 5.0) -> Tuple[List[str], List[float]]:
        """
        Filter documents to keep only those relevant to function calling
        """
        filtered_docs = []
        scores = []
        
        for doc in docs:
            score = self.calculate_function_calling_score(doc)
            if score >= min_score:
                filtered_docs.append(doc)
                scores.append(score)
        
        return filtered_docs, scores

def generate_function_calling_embeddings(sample_size=1000, target_tokens=None, output_path=None, 
                                       min_function_score=5.0, use_function_calling_model=True):
    """
    Loads the ClimbLab dataset, filters for function calling content, and generates specialized embeddings.
    
    Args:
        sample_size: Number of documents to sample initially (used if target_tokens is None)
        target_tokens: Target number of tokens to process (overrides sample_size)
        output_path: Optional path to save the embeddings and texts
        min_function_score: Minimum function calling relevance score to keep documents
        use_function_calling_model: Whether to use a model optimized for code/function calling
    
    Returns:
        tuple: (embeddings_np, texts, function_scores, token_count)
    """
    login()
    
    # Initialize function calling retriever
    retriever = FunctionCallingRetriever()

    # --- 1. Load Model ---
    print("Loading embedding model optimized for function calling...")
    os.system("pip install -q sentence-transformers")
    from sentence_transformers import SentenceTransformer
    
    if use_function_calling_model:
        # Use a code-aware model for better function calling embeddings
        model_name = 'microsoft/codebert-base'
        print(f"Loading code-aware embedding model: {model_name}")
        try:
            model = SentenceTransformer(model_name)
        except:
            # Fallback to general model
            print("Falling back to general embedding model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        print("Loading general embedding model: all-MiniLM-L6-v2")
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- 2. Load and Sample Dataset ---
    dataset = load_dataset("nvidia/ClimbLab", streaming=True, cache_dir="./cache")
    
    if target_tokens:
        print(f"Loading documents to reach target of {target_tokens:,} tokens from nvidia/ClimbLab...")
        sample = []
        total_tokens = 0
        
        for row in dataset["train"]:
            # Count tokens in this document
            if "tokens" in row:
                doc_tokens = len(row["tokens"])
            else:
                # Fallback: estimate tokens as roughly 0.75 * word count
                text = row.get("text", "")
                doc_tokens = int(len(text.split()) * 0.75)
            
            sample.append(row)
            total_tokens += doc_tokens
            
            if total_tokens >= target_tokens:
                print(f"Reached target tokens: {total_tokens:,} tokens with {len(sample)} documents")
                break
    else:
        print(f"Loading and sampling {sample_size} items from nvidia/ClimbLab...")
        sample = list(dataset["train"].take(sample_size))

    # Convert token arrays to text for processing
    docs = []
    for row in sample:
        if "tokens" in row:
            # Try to decode tokens if available
            try:
                # Use a simple tokenizer for decoding
                from transformers import AutoTokenizer
                temp_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                text = temp_tokenizer.decode(row["tokens"], skip_special_tokens=True)
                docs.append(text)
            except:
                # Fallback: join tokens as string
                docs.append(" ".join(map(str, row["tokens"])))
        else:
            # Use text field if available
            text = row.get("text", "")
            docs.append(text)
    
    print(f"Processed {len(docs)} documents from dataset.")

    # --- 3. Filter for Function Calling Content ---
    print(f"Filtering for function calling content (min_score: {min_function_score})...")
    filtered_docs, function_scores = retriever.filter_function_calling_content(docs, min_function_score)
    
    if not filtered_docs:
        print("⚠️ No documents passed the function calling filter! Lowering threshold...")
        filtered_docs, function_scores = retriever.filter_function_calling_content(docs, min_function_score * 0.5)
    
    print(f"Kept {len(filtered_docs)} documents after function calling filtering")
    print(f"Function calling scores range: {min(function_scores):.2f} - {max(function_scores):.2f}")
    print(f"Average function calling score: {np.mean(function_scores):.2f}")

    # --- 4. Generate Embeddings ---
    print("Generating embeddings for function calling content...")
    embeddings_np = model.encode(filtered_docs, convert_to_numpy=True)
    print(f"Generated embeddings with shape: {embeddings_np.shape}")
    
    # Calculate total token count for filtered documents
    total_tokens = 0
    for doc in filtered_docs:
        # Estimate tokens as roughly 0.75 * word count
        total_tokens += int(len(doc.split()) * 0.75)
    
    print(f"Total tokens in function calling content: {total_tokens:,}")

    # --- 5. Save if requested ---
    if output_path:
        print(f"Saving function calling embeddings and texts to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save embeddings as numpy array
        embeddings_file = output_path.replace('.json', '_function_calling_embeddings.npy')
        np.save(embeddings_file, embeddings_np)
        
        # Save texts and metadata
        with open(output_path, 'w') as f:
            data = {
                'texts': filtered_docs,
                'function_scores': function_scores,
                'token_count': total_tokens,
                'embedding_shape': embeddings_np.shape,
                'embedding_file': embeddings_file,
                'min_function_score': min_function_score,
                'original_doc_count': len(docs),
                'filtered_doc_count': len(filtered_docs),
                'retention_rate': len(filtered_docs) / len(docs) * 100,
                'model_used': model_name if use_function_calling_model else 'all-MiniLM-L6-v2'
            }
            json.dump(data, f, indent=2)
        
        print(f"Function calling embeddings saved to {embeddings_file}")
        print(f"Metadata saved to {output_path}")
        print(f"Retention rate: {len(filtered_docs) / len(docs) * 100:.2f}%")
    
    return embeddings_np, filtered_docs, function_scores, total_tokens

def load_function_calling_embeddings(metadata_path):
    """
    Load previously generated function calling embeddings and texts.
    
    Args:
        metadata_path: Path to the metadata JSON file
    
    Returns:
        tuple: (embeddings_np, texts, function_scores, token_count)
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    embeddings_np = np.load(metadata['embedding_file'])
    texts = metadata['texts']
    function_scores = metadata['function_scores']
    token_count = metadata['token_count']
    
    print(f"Loaded {len(texts)} function calling documents")
    print(f"Retention rate was: {metadata.get('retention_rate', 0):.2f}%")
    
    return embeddings_np, texts, function_scores, token_count

def search_function_calling_content(query: str, embeddings_np: np.ndarray, texts: List[str], 
                                  function_scores: List[float], top_k: int = 5, 
                                  model_name: str = 'all-MiniLM-L6-v2') -> List[Dict[str, Any]]:
    """
    Search for function calling content similar to the query.
    
    Args:
        query: Search query
        embeddings_np: Precomputed embeddings
        texts: Corresponding texts
        function_scores: Function calling relevance scores
        top_k: Number of top results to return
        model_name: Name of the embedding model to use for query encoding
    
    Returns:
        List of dictionaries with search results
    """
    from sentence_transformers import SentenceTransformer
    
    # Load the same model used for document embeddings
    model = SentenceTransformer(model_name)
    
    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Calculate cosine similarities
    similarities = np.dot(embeddings_np, query_embedding.T).flatten()
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Prepare results
    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            'rank': i + 1,
            'text': texts[idx],
            'similarity': float(similarities[idx]),
            'function_score': function_scores[idx],
            'text_length': len(texts[idx]),
            'preview': texts[idx][:200] + "..." if len(texts[idx]) > 200 else texts[idx]
        })
    
    return results

if __name__ == '__main__':
    # Function calling-specific test
    print("🔧 Running Function Calling Retriever Test...")
    print("=" * 60)
    
    embeddings, texts, function_scores, token_count = generate_function_calling_embeddings(
        sample_size=500,  # Using moderate sample for testing
        output_path="data/embeddings/function_calling_embeddings.json",
        min_function_score=3.0,  # Lower threshold for testing
        use_function_calling_model=True
    )
    
    print(f"\n📊 Results:")
    print(f"Generated {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
    print(f"Processed {token_count:,} tokens in function calling content")
    print(f"Function calling scores: {min(function_scores):.2f} - {max(function_scores):.2f}")
    
    # Test search functionality
    if len(texts) > 0:
        print(f"\n🔍 Testing search functionality...")
        test_queries = [
            "call a function",
            "API endpoint",
            "execute command",
            "voice assistant",
            "smart home control"
        ]
        
        for query in test_queries[:2]:  # Test first 2 queries
            print(f"\nQuery: '{query}'")
            results = search_function_calling_content(
                query, embeddings, texts, function_scores, top_k=3
            )
            for result in results:
                print(f"  Rank {result['rank']}: Similarity={result['similarity']:.3f}, "
                      f"Function Score={result['function_score']:.2f}")
                print(f"  Preview: {result['preview']}")
    
    print("\n✅ Function calling retriever test finished!") 