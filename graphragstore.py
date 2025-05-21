import re
from llama_index.core.graph_stores import SimplePropertyGraphStore
import networkx as nx
from graspologic.partition import hierarchical_leiden
import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.llama_cpp import LlamaCPP
import re
from graphragextractor import GraphRAGExtractor
from llama_index.core.prompts import PromptTemplate


# self.max_paths_per_chunk

from llama_index.core.llms import ChatMessage
class GraphRAGStore(SimplePropertyGraphStore):
    def __init__(self, llm):
        super().__init__()
    
        self.community_summary = {}
        self.max_cluster_size = 5
        self.model = llm
        self.extract_prompt = PromptTemplate(
            "Given relationships from a knowledge graph in the form: "
            "entity1 -> entity2 -> relation -> description, write a concise summary. "
            "Include the entity names and key points from the descriptions to explain the nature and importance of each relationship clearly and coherently. "
            "Examples:\n"
            "1.\n"
            "Input: Einstein -> Theory of Relativity -> developed -> Einstein formulated the theory to explain how space and time are linked for objects moving at a constant speed.\n"
            "Output: Einstein developed the Theory of Relativity to explain the connection between space and time for objects in uniform motion.\n\n"
            "2.\n"
            "Input: Apple Inc. -> iPhone -> manufactures -> Apple designs and produces the iPhone, a widely used smartphone that revolutionized mobile technology.\n"
            "Output: Apple Inc. manufactures the iPhone, a groundbreaking smartphone that transformed mobile technology."
        )
    
    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        clean_response = self.model.predict(
            self.extract_prompt,
            text=text,
            max_knowledge_triplets=2,
            )
        
        
        clean_response = re.sub(r"^assistant:\s*", "", str(clean_response)).strip()
        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))
        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """Collect detailed information for each node based on their community."""
        community_mapping = {item.node: item.cluster for item in clusters}
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in community_info:
                community_info[cluster_id] = []

            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail)
        return community_info

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary