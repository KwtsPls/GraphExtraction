import ast
import re
import os
import sys
import argparse
import pandas as pd
import networkx
from utils import *
from networkx import draw, Graph

from pyjedai.evaluation import Evaluation
from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding

def clean_entity(entity):
    # Remove newline, dashes, excessive whitespace
    return re.sub(r'\s*[\n\r-]+\s*', ' ', entity).strip()

def parse_file(filename):
    entities = []
    relationships = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line.startswith('['):
                continue  # skip irrelevant lines

            try:
                data = ast.literal_eval(line)  # safer than eval
            except (ValueError, SyntaxError):
                continue  # skip malformed lines

            for item in data:
                if isinstance(item, tuple):
                    cleaned = tuple(clean_entity(i) if isinstance(i, str) else i for i in item)
                    if len(cleaned) == 3:
                        entities.append(cleaned)
                    elif len(cleaned) == 4:
                        relationships.append(cleaned)

    return entities, relationships

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity matching with PyJedAI.")
    parser.add_argument('--input', required=True, help='Path to the output.log file')
    parser.add_argument('--outputFormat', required=False, help='Output format: [simple=default,ntriples,neo4j]')

    args = parser.parse_args()

    entities, relationships = parse_file(args.input)
    entity_dict={}
    for entity in entities:
        entity_dict[entity[0]]=(entity[1],entity[2])

    unknown_rels=set()
    for r in relationships:
        r1 = entity_dict.get(r[0],-1)
        if r1==-1:
             unknown_rels.add(r[0])
        r2 = entity_dict.get(r[1],-1)
        if r2==-1:
             unknown_rels.add(r[1])


    entities = []
    for e,values in entity_dict.items():
        entities.append((e,values[0],values[1]))

    rels_df = pd.DataFrame(list(unknown_rels), columns =['entity'])
    entities_df = pd.DataFrame(entities, columns =['entity','type','description'])
    entities_df = entities_df[['entity']]

    print(rels_df)
    print(entities_df)

    attr1 = entities_df.columns[1:].to_list()
    attr2 = rels_df.columns[1:].to_list()

    data = Data(dataset_1=entities_df,
                attributes_1=attr1,
                id_column_name_1='entity',
                dataset_2=rels_df,
                attributes_2=attr2,
                id_column_name_2='entity')

    emb = EmbeddingsNNBlockBuilding(vectorizer='sminilm',
                                    similarity_search='faiss')

    blocks, g = emb.build_blocks(data,
                                top_k=1,
                                similarity_distance='cosine',
                                load_embeddings_if_exist=False,
                                save_embeddings=False,
                                with_entity_matching=True)

    mapping_df = emb.export_to_df(blocks)
    merged1 = pd.merge(rels_df, mapping_df, left_on='entity', right_on='id1', how='inner')
    final_df = pd.merge(merged1, entities_df, left_on='id2', right_on='entity', how='inner', suffixes=('_re', '_en'))
    final_df = final_df.drop(columns=['id1', 'id2'])

    # --- Save output ---
    final_df.to_csv("mappings.csv", index=False)

    mapping_dict={}
    for index,row in final_df.iterrows():
        mapping_dict[row['entity_re']] = row['entity_en']
        unknown_rels.remove(row['entity_re'])
    
    corrected_rels=[]
    for r in relationships:
        source = r[0]
        target = r[1]
        if(mapping_dict.get(source,-1)!=-1):
            source = mapping_dict[source]
        if(mapping_dict.get(target,-1)!=-1):
            target = mapping_dict[target]
        corrected_rels.append((source,target,r[2],r[3]))
    relationships = corrected_rels

    for item in unknown_rels:
        entities.append((item,'Resource',''))

    if args.outputFormat == "ntriples":
        export_neo4j(entities,relationships)
    elif args.outputFormat == "ntriples":
        export_ntriples(entities,relationships)
    else:
        export(entities,relationships)
    
