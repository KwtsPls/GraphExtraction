import re
import os
import pandas as pd
import urllib.parse

def export(entities,relationships,entities_file="entities.csv", rels_file="relationships.csv"):
    rels_df = pd.DataFrame(relationships, columns =['source','target','relationship','originalText'])
    entities_df = pd.DataFrame(entities, columns =['entity','type','description'])
    rels_df.to_csv(rels_file, index=False)
    entities_df.to_csv(entities_file, index=False)

def export_ntriples(entities, relationships, filename="data.nt", 
                          entity_base_uri="http://graphrag.example.com/resource/", 
                          ontology_base_uri="http://graphrag.example.com/ontology/"):
    """
    Create a .nt file from entity and relationship DataFrames.

    Parameters:
        relationships (list of tuples): List of the form [('source','target','relationship','originalText')])
        entities_df (list of tuples):  List of the form [('entity','type','description')]
        filename (str): Output filename for the .nt file
        entity_base_uri (str): Base URI for entities
        ontology_base_uri (str): Base URI for types and relationships
    """
    rels_df = pd.DataFrame(relationships, columns =['source','target','relationship','originalText'])
    entities_df = pd.DataFrame(entities, columns =['entity','type','description'])

    triples = []


    def sanitize_uri_component(component):
        return urllib.parse.quote(component.replace(" ", "_"), safe="")

    # Create triples for entities
    for _, row in entities_df.iterrows():
        entity_uri = f"<{entity_base_uri}{sanitize_uri_component(row['entity'])}>"
        type_uri = f"<{ontology_base_uri}{sanitize_uri_component(row['type'])}>"
        desc_literal = f'"{row["description"]}"'
        desc_literal = desc_literal.replace("\"", "'")

        # rdf:type triple
        triples.append(f"{entity_uri} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> {type_uri} .")

        # custom description triple
        triples.append(f"{entity_uri} <{ontology_base_uri}description> {desc_literal} .")

    # Create triples for relationships
    for _, row in rels_df.iterrows():
        source_uri = f"<{entity_base_uri}{sanitize_uri_component(row['source'])}>"
        target_uri = f"<{entity_base_uri}{sanitize_uri_component(row['target'])}>"
        relation_uri = f"<{ontology_base_uri}{sanitize_uri_component(row['relationship'])}>"

        triples.append(f"{source_uri} {relation_uri} {target_uri} .")

    # Write to .nt file
    with open(filename, 'w', encoding='utf-8') as f:
        for triple in triples:
            f.write(triple + '\n')


def export_neo4j(entities, relationships, nodes_file="nodes.csv", rels_file="relationships.csv"):
    """
    Create CSV files for Neo4j admin import tool from entity and relationship DataFrames.

    Parameters:
        rels_df (pd.DataFrame): DataFrame with columns ['source','target','relationship','originalText']
        entities_df (pd.DataFrame): DataFrame with columns ['entity','type','description']
        nodes_file (str): Output filename for the nodes CSV
        rels_file (str): Output filename for the relationships CSV
    """
    # Nodes CSV
    rels_df = pd.DataFrame(relationships, columns =['source','target','relationship','originalText'])
    entities_df = pd.DataFrame(entities, columns =['entity','type','description'])
    nodes_df = entities_df.rename(columns={
        'entity': 'id:ID',
        'type': ':LABEL',
        'description': 'description'
    })
    nodes_df.to_csv(nodes_file, index=False)

    # Relationships CSV
    rels_df_clean = rels_df.rename(columns={
        'source': ':START_ID',
        'target': ':END_ID',
        'relationship': ':TYPE'
    })
    rels_df_clean = rels_df_clean[[':START_ID', ':END_ID', ':TYPE']]
    rels_df_clean.to_csv(rels_file, index=False)