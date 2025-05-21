import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.llama_cpp import LlamaCPP
import re
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from graphragextractor import GraphRAGExtractor
from graphragstore import GraphRAGStore
from llama_index.core import PropertyGraphIndex
from llama_index.core import Settings

# Load sample da`taset
news = pd.read_csv("incidents.csv", sep=',')


columns = ["description","product","hazard","productCategory","hazardCategory","supplier"]

# Convert data into LlamaIndex Document objects
documents = [
    Document(text=f"{row['originalTitle']}: {row['description']}")
    for _, row in news.iterrows()
]


splitter = SentenceSplitter(
    chunk_size=4096,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)



llm = LlamaCPP(
    model_path="./models/mistral-7b-instruct-v0.3.Q4_K_M.gguf",  # local GGUF path
    temperature=0.1,
    max_new_tokens=4096,
    context_window=4096,
    model_kwargs={
        "n_gpu_layers": 100,        # adjust based on VRAM (e.g. 30-50 layers for 8GB)
        "n_ctx": 4096,
        "n_threads": 8,            # or more if you have more CPU cores
    },
    verbose=True,
)

entity_pattern = r"entity_name:\s*(.*?)\s*entity_type:\s*(.*?)\s*entity_description:\s*(.*?)(?=\n|$)"
relationship_pattern = r"source_entity:\s*(.*?)\s*target_entity:\s*(.*?)\s*relation:\s*(.*?)\s*relationship_description:\s*(.*?)(?=\n|$)"

def parse_fn(response_str: str):
    entities = re.findall(entity_pattern, response_str, flags=re.DOTALL)
    relationships = re.findall(relationship_pattern, response_str, flags=re.DOTALL)
    return entities, relationships


KG_TRIPLET_EXTRACT_TMPL = """
Extract entities and their relationships from the text below.

For each entity, provide:
- entity_name: (capitalize it)
- entity_type: (one or two words)
- entity_description: (short summary)

For each relationship between two entities, provide:
- source_entity: (name of the source entity)
- target_entity: (name of the target entity)
- relation: (short label)
- relationship_description: (brief reason for the relation)

TEXT:
####################
{text}
####################

OUTPUT FORMAT:
First list all entities, then list all relationships.

Example:

entity_name: Apple
entity_type: Company
entity_description: A technology company known for iPhones.

entity_name: iPhone
entity_type: Product
entity_description: A smartphone designed and sold by Apple.

source_entity: Apple
target_entity: iPhone
relation: Produces
relationship_description: Apple designs and sells the iPhone.

(Your output starts here:)
"""

kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=2,
    parse_fn=parse_fn,
)


Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # or any other HF embedding model
)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = PropertyGraphIndex(
    nodes=nodes,
    property_graph_store=GraphRAGStore(llm = llm),
    kg_extractors=[kg_extractor],
    embed_model=embed_model,      
    show_progress=True,
)