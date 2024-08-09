DIR = "../data/csvs/"
RES = "../results/"

import  argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset ID")
parser.add_argument("--schema", type=str, default=None, help="Dataset ID")

args = parser.parse_args()

dataset = args.dataset

#  ---------------- PLOTS CODE ---------------------- #

from networkx.algorithms import bipartite
import sys, os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


from strsimpy.sorensen_dice import SorensenDice
ids_in_gt = dict()
ids_in_gt_reversed = dict()

def weight_distribution(G):
  bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  distribution = [0] * (len(bins)-1)
  for u, v, w in G.edges(data='weight'):
    for i in range(len(bins) - 1):
        if bins[i] <= w < bins[i + 1]:
            distribution[i] += 1
            break
  return distribution, len(G.edges(data='weight'))


def plot_distribution(G, title='From maybe matches to exact similar'):
  labels = [f'{(i)/10:.1f} - {(i+1)/10:.1f}' for i in range(5, 10)]

  distribution, num_of_pairs = weight_distribution(G)
  x = np.arange(len(labels)) 
  width = 0.35

  distribution = list(map(lambda x: x, distribution))

  print("Number of pairs: ", distribution)
  fig, ax = plt.subplots(figsize=(10,6))
  r1 = ax.plot(x, distribution, color='blue', marker='*', markersize=15)
  symb = [r'$\sim$',r'$\simeq$',r'$\approx$',r'=',r'$\equiv$']
  ax.set_xticks(x)
  ax.set_xticklabels(symb, fontsize=16)
  ax.set_ylabel('Number of pairs')
  ax.set_title(title)
  ax.set_xlabel('Similarity score ranges')
  fig.tight_layout()
  plt.savefig(RES+f'similarity_distributions_blocking_1_{dataset}.png')
  plt.show()

#  ---------------- PYJEDAI CODE ---------------------- #

import os
import sys
import pandas as pd
import networkx
from networkx import draw, Graph
from pyjedai.utils import (
    text_cleaning_method,
    print_clusters,
    print_blocks,
    print_candidate_pairs
)
from pyjedai.evaluation import Evaluation
from pyjedai.datamodel import Data
import time
import warnings
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import UniqueMappingClustering
from pyjedai.matching import EntityMatching

warnings.simplefilter(action='ignore', category=FutureWarning)


# schema = args.schema

dnames = [('Abt', 'Buy'), ('Wallmart', 'Amazon'), ('Amazon', 'GoogleProducts')]
datasetsD1 = ["D2abt.csv", "D3amazon.csv", "D8walmart.csv"]
datasetsD2 = ["D2buy.csv", "D3gp.csv",  "D8amazon.csv"]
groundtruthDirs = ["D2gt.csv", "D3gt.csv", "D8gt.csv"]

separator = [
     '|', '#', '|'
]

if dataset == 'Abt - Buy':
  datasetId = 0
elif dataset == 'Wallmart - Amazon':
  datasetId = 1
elif dataset == 'Amazon - Google Products':
  datasetId = 2

dmi = datasetId

print(u'\u2500' * 123)
print(f"Dataset: {dnames[dmi][0]} - {dnames[dmi][1]}")
print(u'\u2500' * 123)
d1 = pd.read_csv(DIR+datasetsD1[dmi], sep=separator[dmi], na_filter=False).astype(str)
d2 = pd.read_csv(DIR+datasetsD2[dmi], sep=separator[dmi], na_filter=False).astype(str)
gt = pd.read_csv(DIR+groundtruthDirs[dmi], sep=separator[dmi]).astype(str)

attr1 = d1.columns[1:].to_list()
attr2 = d2.columns[1:].to_list()

attr1.remove('aggregate value')
attr2.remove('aggregate value')

data = Data(dataset_1=d1,
            id_column_name_1='id',
            dataset_name_1=dnames[dmi][0],
            dataset_2=d2,
            id_column_name_2='id',
            dataset_name_2=dnames[dmi][1],
            ground_truth=gt)

from pyjedai.block_building import StandardBlocking
from pyjedai.block_cleaning import BlockFiltering
from pyjedai.comparison_cleaning import WeightedEdgePruning, WeightedNodePruning
from pyjedai.matching import EntityMatching

if dmi == 0:
    bb = StandardBlocking()
    blocks = bb.build_blocks(data)

    bf = BlockFiltering(ratio=0.9)
    filtered_blocks = bf.process(blocks, data, tqdm_disable=False)

    mb = WeightedEdgePruning(weighting_scheme='EJS')
    candidate_pairs_blocks = mb.process(filtered_blocks, data, tqdm_disable=True)

    em = EntityMatching(
        metric='cosine',
        tokenizer='char_tokenizer',
        vectorizer='tfidf',
        qgram=3,
        similarity_threshold=0.0
    )

    pairs_graph = em.predict(candidate_pairs_blocks, data, tqdm_disable=True)

    ccc = UniqueMappingClustering()
    clusters = ccc.process(pairs_graph, data, similarity_threshold=0.25)
    _ = ccc.evaluate(clusters, with_classification_report=True)

    plot_distribution(pairs_graph, 'Vector based blocking')

elif dmi ==1:

    bb = StandardBlocking()
    blocks = bb.build_blocks(data)

    bf = BlockFiltering(ratio=0.075)
    filtered_blocks = bf.process(blocks, data, tqdm_disable=False)

    mb = WeightedEdgePruning(weighting_scheme='EJS')
    candidate_pairs_blocks = mb.process(filtered_blocks, data, tqdm_disable=True)

    em = EntityMatching(
        metric='cosine',
        tokenizer='char_tokenizer',
        vectorizer='tfidf',
        qgram=2,
        similarity_threshold=0.0
    )

    pairs_graph = em.predict(candidate_pairs_blocks, data, tqdm_disable=True)

    ccc = UniqueMappingClustering()
    clusters = ccc.process(pairs_graph, data, similarity_threshold=0.5)
    _ = ccc.evaluate(clusters, with_classification_report=True)

    plot_distribution(pairs_graph, 'Vector based blocking')

else:
    bb = StandardBlocking()
    blocks = bb.build_blocks(data)

    bf = BlockFiltering(ratio=0.6)
    filtered_blocks = bf.process(blocks, data, tqdm_disable=False)

    mb = WeightedNodePruning(weighting_scheme='CBS')
    candidate_pairs_blocks = mb.process(filtered_blocks, data, tqdm_disable=True)

    em = EntityMatching(
        metric='cosine',
        tokenizer='word_tokenizer',
        vectorizer='tfidf',
        qgram=2,
        similarity_threshold=0.0
    )

    pairs_graph = em.predict(candidate_pairs_blocks, data, tqdm_disable=True)

    ccc = UniqueMappingClustering()
    clusters = ccc.process(pairs_graph, data, similarity_threshold=0.1)
    _ = ccc.evaluate(clusters, with_classification_report=True)

    plot_distribution(pairs_graph, 'Vector based blocking')

def get_gt_scores_only(all_pairs, data):

  gt_pairs = {}
  for _, (id1, id2) in data.ground_truth.iterrows():

    id1 = int(data._ids_mapping_1[id1])
    id2 = int(data._ids_mapping_2[id2])

    if (id1,id2) in all_pairs:
      gt_pairs[(id1,id2)] = all_pairs[(id1,id2)]

  return gt_pairs


def sim_graph_to_pairs(graph, dataset_limit):

  pairs = {}
  for u, v, w in graph.edges(data='weight'):
    if u < dataset_limit:
      id1=u
      id2=v
    else:
      id1=v
      id2=u

    pairs[(id1,id2)] = w

  return pairs

sims = sim_graph_to_pairs(pairs_graph, data.dataset_limit)
gt_pairs = get_gt_scores_only(sims, data)

matching_pairs = gt_pairs
matching_scores = [sims[(e1, e2)] for e1, e2 in sims if (e1, e2) in matching_pairs]
non_matching_scores = [sims[(e1, e2)] for e1, e2 in sims if (e1, e2) not in matching_pairs]


def plot_matching_distributions(matching_scores, non_matching_scores, bins=0.01):
    max_score = max(max(matching_scores), max(non_matching_scores))

    bins = np.arange(0, max_score + bins, bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    ax1.hist(matching_scores, bins=bins, edgecolor='blue', alpha=0.7)
    ax1.set_title('Similarity Distribution of Matching Pairs')
    ax1.set_xlabel('Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')

    ax2.hist(non_matching_scores, bins=bins, edgecolor='blue', alpha=0.7)
    ax2.set_title('Similarity Distribution of Non-Matching Pairs')
    ax2.set_xlabel('Similarity')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(RES+f'similarity_distributions_blocking_2_{dataset}.png')
    plt.show()

plot_matching_distributions(matching_scores, non_matching_scores)