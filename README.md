# Overview

An LLM-powered approach was developed to generate graphs and knowledge graphs directly from textual data, serving as the foundational phase of the GraphRAG pipeline. This system leverages the capabilities of the llama-index framework and employs a property graph model to structurally represent extracted information. At its core lies a custom GraphRAGExtractor, which uses a large language model (LLM) to process natural language input and extract semantic triples—entity-relation-entity structures—along with entity and relationship metadata. This is achieved through prompt engineering and asynchronous parallelization to efficiently scale over document chunks. Each node in the graph is enriched with metadata, and relationships are described with contextual summaries to enhance graph interpretability. The resulting property graph is further processed using community detection techniques, such as hierarchical Leiden clustering, to reveal structural patterns. Summaries of each community are generated using another LLM prompt-driven process, resulting in coherent explanations of intra-community relationships.

# Install

pyJedAI has been tested in Windows and Linux OS. 

__Basic requirements:__

- Python version greater or equal to **3.10**.

Set up locally:
```
git clone https://github.com/AI-team-UoA/pyJedAI.git
```
go to the root directory with `cd GraphExtraction` and type:
```
pip install -r requirements.txt
```
# 🔍 Basic Usage: LLM-Powered Knowledge Graph Extraction from Text

An example of how to use graphragextractor.py and graphragstore.py are implemented in graphrag.py. In this example llama-index is combined with Mistral-7B using llama.cpp as a basis. The provided pipeline works as follows:

Running the following command:

    python grapgrag.py

Using test data from [Jason Brownlee's implementation](https://machinelearningmastery.com/building-graph-rag-system-step-by-step-approach/) tutorial

1. 📄 Extract Entities and Relationships
Generate a property graph from raw text by identifying and linking named entities using LLM prompts.

2. 🧠 Summarize Semantic Communities
Apply hierarchical Leiden clustering to group related nodes and edges, then use an LLM to generate coherent summaries for each community.

Running the following command:

    python parser.py output.log [simple/neo4j/ntriples]

Users can extract the entities and relationhsips after they are cleaned and matched using [PyJedAI](https://github.com/AI-team-UoA/pyJedAI/), in various forms:

  - simple: a simple mapping of entities and relationships to .csv
  - neo4j: .csv files that can be directly imported in a neo4j dataset
  - ntriples: a .nt file format for usage in semantic web settings

Users can create python scripts similar to graphgrap.py to utilize their own LLMs and source files for graph extraction from raw text.

# Notebook Demo

A tutorial in the form of a .ipynb notebook is also presented in the tutorial subfolder.

# License

Released under the Apache-2.0 license (see [LICENSE.txt](https://github.com/AI-team-UoA/pyJedAI/blob/main/LICENSE)).

Copyright © 2025 AI-Team, University of Athens

<div align="center">
    <hr>
    <br>
    <a href="https://stelar-project.eu">
        <img align="center" src="https://stelar-project.eu/wp-content/uploads/2022/08/Logo-Stelar-1-f.png" width=180/>
    </a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://ec.europa.eu/info/index_en">
        <img align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/1200px-Flag_of_Europe.svg.png" width=140/>
    </a>
    <br>
    <br>
        <b>This project is being funded in the context of <a href="https://stelar-project.eu">STELAR</a> that is an <a href="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en">HORIZON-Europe</a> project.
        </b>
    <br>
</div>
