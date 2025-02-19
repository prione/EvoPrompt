from sentence_transformers import SentenceTransformer
import faiss
import config

emb = SentenceTransformer(config.emb_model)
dimension = emb.get_sentence_embedding_dimension()

def evaluate(genotypes:list, phenotypes:list, expected:str):
    index = faiss.IndexFlatIP(dimension)
    
    phenotypes = ["query: " + ans for ans in phenotypes]
    embeddings = emb.encode(phenotypes)
    expected_embeddings = emb.encode(["query: " + expected])
    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(expected_embeddings)
    
    index.add(embeddings)
    similarities, indices = index.search(expected_embeddings, config.genotypes_num)
    
    genotypes = [genotypes[i] for i in indices[0] if i!=-1]
    phenotypes = [phenotypes[i] for i in indices[0] if i!=-1 ]
    similarities = [s for s, i in zip(similarities[0], indices[0]) if i!=-1]
    
    return genotypes, phenotypes, similarities