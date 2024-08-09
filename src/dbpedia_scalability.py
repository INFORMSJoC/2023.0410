import plotly.express as px
import os
import pandas as pd

d1=pd.read_csv('../data/dbpedia/newDBPedia1.csv', sep='|')
d2=pd.read_csv('../data/dbpedia/newDBPedia2.csv', sep='|')
gt=pd.read_csv('../data/dbpedia/newDBPediaMatchesgt.csv', header=None, sep='|')

from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import UniqueMappingClustering

# -------------------------------------- #

LANGUAGE_MODEL = "sminilm"
TOPK = 10
CLUSTERING = UniqueMappingClustering
THRESHOLD = 0.8

# -------------------------------------- #

data = Data(dataset_1=d1,
            id_column_name_1='Id',
            dataset_2=d2,
            id_column_name_2='Id',
            ground_truth=gt)

emb = EmbeddingsNNBlockBuilding(vectorizer=LANGUAGE_MODEL,
                                similarity_search='faiss')

blocks, g = emb.build_blocks(data,
                             top_k=TOPK,
                             similarity_distance='euclidean',
                             load_embeddings_if_exist=True,
                             save_embeddings=True,
                             with_entity_matching=True)

ccc = CLUSTERING()
clusters = ccc.process(g, data, similarity_threshold=THRESHOLD)
nn_pairs_df = ccc.export_to_df(clusters)
