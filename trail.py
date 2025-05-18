import os
import re
import json
import time
import uuid
import pinecone
import openai
from typing import List
from dotenv import load_dotenv

load_dotenv()
openai.api_key = ""
PINECONE_API_KEY = ""
PINECONE_ENVIRONMENT = "us-east1-gcp"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "scraped-content"

# Model parameters
EMBEDDING_MODEL = "text-embedding-3-small"  
EMBEDDING_DIMENSION = 1536  
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 200  
MAX_CHUNKS_PER_BATCH = 100  

def initialize_pinecone():
    """Initialize Pinecone client and ensure index exists"""
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine"
        )
        time.sleep(30)
    return pc.Index(PINECONE_INDEX_NAME)

def clean_and_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Clean and split text into chunks of specified size with overlap"""
    if not text or not isinstance(text, str):
        return []
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end < text_length:
            if text[end-1] not in ".,!? ":
                last_period = text.rfind('.', start, end)
                last_space = text.rfind(' ', start, end)
                if last_period > start + (chunk_size // 2):
                    end = last_period + 1
                elif last_space > start + (chunk_size // 2):
                    end = last_space + 1
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

def get_embeddings(chunks: List[str]) -> List[List[float]]:
    """Get embeddings for chunks using OpenAI's embedding API"""
    embeddings = []
    for i in range(0, len(chunks), MAX_CHUNKS_PER_BATCH):
        batch = chunks[i:i + MAX_CHUNKS_PER_BATCH]
        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            if i + MAX_CHUNKS_PER_BATCH < len(chunks):
                time.sleep(0.5) 
                
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            embeddings.extend([[0] * EMBEDDING_DIMENSION] * len(batch))
    
    return embeddings


def process_json_file(file_path: str):
    """Process scraped content JSON file and ingest into Pinecone"""
    print(f"Loading scraped content from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize Pinecone
    index = initialize_pinecone()
    
    # Extract and process original article
    print("Processing original article...")
    article_text = data.get('original_text', '')
    article_chunks = clean_and_chunk_text(article_text)
    
    all_chunks = []
    all_metadata = []
    
    # Process original article chunks
    for i, chunk in enumerate(article_chunks):
        chunk_id = f"original_article_{i}"
        all_chunks.append(chunk)
        all_metadata.append({
            "id": chunk_id,
            "type": "original_article",
            "chunk_index": i,
            "total_chunks": len(article_chunks),
            "source": "original_article"
        })
    
    scraped_content = data.get('scraped_content', {})
    print(f"Processing {len(scraped_content)} scraped pages...")
    
    for page_id, page_data in scraped_content.items():
        if page_data.get('status') != 'success':
            continue
        
        # Get page content
        url = page_data.get('url', '')
        title = page_data.get('title', '')
        text_content = page_data.get('text_content', '')
        
        # Skip if no substantial content
        if len(text_content) < 50:
            continue
        
        # Chunk page content
        page_chunks = clean_and_chunk_text(text_content)
        
        # Process page chunks
        for i, chunk in enumerate(page_chunks):
            chunk_id = f"page_{page_id}_chunk_{i}"
            all_chunks.append(chunk)
            all_metadata.append({
                "id": chunk_id,
                "type": "scraped_page",
                "url": url,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(page_chunks),
                "page_id": page_id
            })
            
         
            if i == 0 and page_data.get('links'):  # Only store links with first chunk
                link_texts = []
                for link in page_data.get('links', []):
                    link_url = link.get('url', '')
                    link_text = link.get('text', '')
                    if link_url and link_text:
                        link_texts.append(f"{link_text}: {link_url}")
                
                if link_texts:
                    links_text = "Links found on page:\n" + "\n".join(link_texts)
                    links_chunks = clean_and_chunk_text(links_text)
                    
                    for j, link_chunk in enumerate(links_chunks):
                        chunk_id = f"page_{page_id}_links_{j}"
                        all_chunks.append(link_chunk)
                        all_metadata.append({
                            "id": chunk_id,
                            "type": "page_links",
                            "url": url,
                            "title": title + " - Links",
                            "chunk_index": j,
                            "total_chunks": len(links_chunks),
                            "page_id": page_id
                        })
    
    # Get embeddings for all chunks
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = get_embeddings(all_chunks)
    
    # Prepare vectors for Pinecone
    vectors = []
    for i, (chunk, embedding, metadata) in enumerate(zip(all_chunks, embeddings, all_metadata)):
        vector_id = str(uuid.uuid4())
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                **metadata,
                "text": chunk[:1000],  # Store truncated text in metadata
                "timestamp": time.time()
            }
        })
        
        # Upsert in batches of 100
        if len(vectors) >= 100 or i == len(all_chunks) - 1:
            print(f"Upserting batch of {len(vectors)} vectors to Pinecone...")
            index.upsert(vectors=vectors)
            vectors = []
    
    print("Ingestion complete!")
    return len(all_chunks)

def query_similar_content(query_text: str, top_k: int = 5):
    """Query Pinecone for similar content based on the query text"""
    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Get query embedding
    query_embedding = get_embeddings([query_text])[0]
    
    # Query Pinecone
    query_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Process and return results
    results = []
    for match in query_results.matches:
        results.append({
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("url", match.metadata.get("source", "")),
            "title": match.metadata.get("title", "")
        })
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest scraped content into Pinecone')
    parser.add_argument('--file', required=True, help='Path to scraped_content.json file')
    parser.add_argument('--query', help='Query to search for similar content')
    
    args = parser.parse_args()
    
    if args.file and not args.query:
        # Ingest content
        chunks_processed = process_json_file(args.file)
        print(f"Successfully processed {chunks_processed} chunks")
    elif args.query:
        # Search for similar content
        results = query_similar_content(args.query)
        print("\nSearch Results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['title']} (Score: {result['score']:.4f})")
            print(f"Source: {result['source']}")
            print(f"Content: {result['text'][:200]}...")
    else:
        parser.print_help() 