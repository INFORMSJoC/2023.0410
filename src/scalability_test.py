
import pandas as pd
import os
import plotly.express as px

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

import jnius_config
jnius_config.add_classpath('jedai-core-with-joins.jar')
from jnius import autoclass

#@title Visualization of the similarities distribution
from networkx.algorithms import bipartite
import sys, os
import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx

# !pip install strsimpy
from strsimpy.sorensen_dice import SorensenDice
ids_in_gt = dict()
ids_in_gt_reversed = dict()

# def get_gt_scores_only(data):

#   bins_g = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#   distribution_g = [0] * (len(bins_g)-1)

#   bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#   distribution = [0] * (len(bins)-1)

#   for _, (id1, id2) in data.ground_truth.iterrows():
#     id1 = data._ids_mapping_1[id1]
#     id2 = data._ids_mapping_2[id2]
#     ids_in_gt[id1] = id2
#     ids_in_gt_reversed[id2] = id1

#     similarity = SorensenDice().distance(
#       data.entities.iloc[id1].str.cat(sep=' '),
#       data.entities.iloc[id2].str.cat(sep=' ')
#     )

#     for i in range(len(bins) - 1):
#       if bins[i] <= similarity < bins[i + 1]:
#           distribution[i] += 1
#           break
#   return distribution, data.ground_truth.shape[0]

def weight_distribution(G):
  bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  distribution = [0] * (len(bins)-1)
  for u, v, w in G.edges(data='weight'):
    # print(u, v)
    # if (u in ids_in_gt and ids_in_gt[u]!=v) or (v in ids_in_gt and ids_in_gt[v]!=u) or (u in ids_in_gt_reversed and ids_in_gt_reversed[u]!=v) or (v in ids_in_gt_reversed and ids_in_gt_reversed[v]!=u):
    for i in range(len(bins) - 1):
        if bins[i] <= w < bins[i + 1]:
            distribution[i] += 1
            break
  return distribution, len(G.edges(data='weight'))

import numpy as np
import networkx as nx

def plot_distribution(G, title='From maybe matches to exact similar'):
  labels = [f'{(i)/10:.1f} - {(i+1)/10:.1f}' for i in range(5, 10)]

  distribution, num_of_pairs = weight_distribution(G)
  # print(distribution)
  x = np.arange(len(labels))  # the label locations
  # print(x)
  width = 0.35  # the width of the bars

  distribution = list(map(lambda x: x, distribution))
  # print("Distribution-% of predicted scores: ", distribution)

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
  plt.show()

BasicConfigurator =  autoclass('org.apache.log4j.BasicConfigurator')
IBlockBuilding =  autoclass('org.scify.jedai.blockbuilding.IBlockBuilding')
StandardBlocking = autoclass('org.scify.jedai.blockbuilding.StandardBlocking')
CardinalityEdgePruning = autoclass('org.scify.jedai.blockprocessing.comparisoncleaning.CardinalityEdgePruning')
GtCSVReader = autoclass('org.scify.jedai.datareader.groundtruthreader.GtCSVReader')
BlocksPerformanceWriter = autoclass('org.scify.jedai.datawriter.BlocksPerformanceWriter')
BlocksPerformance = autoclass('org.scify.jedai.utilities.BlocksPerformance')
AbstractDuplicatePropagation = autoclass('org.scify.jedai.utilities.datastructures.AbstractDuplicatePropagation')
UnilateralDuplicatePropagation = autoclass('org.scify.jedai.utilities.datastructures.UnilateralDuplicatePropagation')
BilateralDuplicatePropagation = autoclass('org.scify.jedai.utilities.datastructures.BilateralDuplicatePropagation')
StringBuilder  = autoclass('java.lang.StringBuilder')
File = autoclass('java.io.File')
PrintWriter = autoclass('java.io.PrintWriter')
List = autoclass('java.util.List')
Set = autoclass('java.util.Set')
ComparisonCleaningMethod = autoclass('org.scify.jedai.utilities.enumerations.ComparisonCleaningMethod')
EntitySerializationReader = autoclass('org.scify.jedai.datareader.entityreader.EntitySerializationReader')
GtSerializationReader = autoclass('org.scify.jedai.datareader.groundtruthreader.GtSerializationReader')
BilateralDuplicatePropagation = autoclass('org.scify.jedai.utilities.datastructures.BilateralDuplicatePropagation')
ComparisonsBasedBlockPurging = autoclass('org.scify.jedai.blockprocessing.blockcleaning.ComparisonsBasedBlockPurging')
BlockFiltering = autoclass('org.scify.jedai.blockprocessing.blockcleaning.BlockFiltering')
ExtendedQGramsBlocking = autoclass('org.scify.jedai.blockbuilding.ExtendedQGramsBlocking')
QGramsBlocking = autoclass('org.scify.jedai.blockbuilding.QGramsBlocking')
SuffixArraysBlocking = autoclass('org.scify.jedai.blockbuilding.SuffixArraysBlocking')
ExtendedSuffixArraysBlocking = autoclass('org.scify.jedai.blockbuilding.ExtendedSuffixArraysBlocking')
BlocksPerformanceWriter = autoclass('org.scify.jedai.datawriter.BlocksPerformanceWriter')
WeightedEdgePruning = autoclass('org.scify.jedai.blockprocessing.comparisoncleaning.WeightedEdgePruning')
WeightingScheme = autoclass('org.scify.jedai.utilities.enumerations.WeightingScheme')

WeightedNodePruning = autoclass('org.scify.jedai.blockprocessing.comparisoncleaning.WeightedNodePruning')
SimilarityPairs = autoclass('org.scify.jedai.datamodel.SimilarityPairs')
UniqueMappingClustering = autoclass('org.scify.jedai.entityclustering.UniqueMappingClustering')
RepresentationModel = autoclass('org.scify.jedai.utilities.enumerations.RepresentationModel')
ClustersPerformance = autoclass('org.scify.jedai.utilities.ClustersPerformance')
ProfileMatcher = autoclass('org.scify.jedai.entitymatching.ProfileMatcher')
SimilarityMetric = autoclass('org.scify.jedai.utilities.enumerations.SimilarityMetric')


import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
from pyjedai.evaluation import Evaluation, write
from pyjedai.datamodel import Data
import time
import warnings
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import UniqueMappingClustering
from pyjedai.matching import EntityMatching

warnings.simplefilter(action='ignore', category=FutureWarning)
# dataset = "Wallmart - Amazon" #@param ["Abt - Buy", "Wallmart - Amazon", "Amazon - Google Products"]
schema_type = "schema-based" #@param ["schema-based", "schema-agnostic"]

import argparse 

def parse_args():
    parser = argparse.ArgumentParser(description='Blocking algorithms')
    parser.add_argument('--dataset', type=str, default='Abt - Buy', help='Dataset to use')
    return parser.parse_args()

args = parse_args()
dataset = args.dataset

if schema_type == "schema-agnostic":
  schema_id = 0
else:
  schema_id = 1

dnames = [('Abt', 'Buy'), ('Amazon', 'GoogleProducts'), ('Wallmart', 'Amazon')]
datasetsD1 = ["D2abt.csv", "D3amazon.csv", "D8walmart.csv"]
datasetsD2 = ["D2buy.csv", "D3gp.csv",  "D8amazon.csv"]
groundtruthDirs = ["D2gt.csv", "D3gt.csv", "D8gt.csv"]

separator = [
     '|', '#', '|'
]

thresholds = [
    [ 0.0, 0.45, 0.87], [0.5, 0.49, 0.82]
]
schema_based_attributes = ["name", "title", "title"]

if dataset == 'Abt - Buy':
  datasetId = 0
elif dataset == 'Amazon - Google Products':
  datasetId = 1
elif dataset == 'Wallmart - Amazon':
  datasetId = 2

dmi = datasetId
print(u'\u2500' * 123)
d1 = pd.read_csv(datasetsD1[dmi], sep=separator[dmi], na_filter=False).astype(str)
d2 = pd.read_csv(datasetsD2[dmi], sep=separator[dmi], na_filter=False).astype(str)
gt = pd.read_csv(groundtruthDirs[dmi], sep=separator[dmi]).astype(str)

print(u'\u2500' * 123)

# d1 = d1[['id','title','description','manufacturer','price']]
# d2 = d2[['id','title','description','manufacturer','price']]

attr1 = d1.columns[1:].to_list()
attr2 = d2.columns[1:].to_list()

attr1.remove('aggregate value')
attr2.remove('aggregate value')

data = Data(dataset_1=d1,
            id_column_name_1='id',
            attributes_1=attr1 if schema_id == 0 else [schema_based_attributes[dmi]],
            dataset_name_1=dnames[dmi][0],
            dataset_2=d2,
            attributes_2=attr2 if schema_id == 0 else [schema_based_attributes[dmi]],
            id_column_name_2='id',
            dataset_name_2=dnames[dmi][1],
            ground_truth=gt)

emb = EmbeddingsNNBlockBuilding(vectorizer='sminilm',
                                similarity_search='faiss')
blocks, g = emb.build_blocks(data,
                            top_k=10,
                            load_embeddings_if_exist=False,
                            save_embeddings=True,
                            with_entity_matching=True)

ccc = UniqueMappingClustering()
clusters = ccc.process(g, data, similarity_threshold=thresholds[schema_id][dmi])
_ = ccc.evaluate(clusters, with_classification_report=True)

import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"RSS: {memory_info.rss / (1024 * 1024 * 1024)} MB")
print(f"VMS: {memory_info.vms / (1024 * 1024 * 1024)} MB")

import psutil
import os

# CPU Memory Usage
process = psutil.Process(os.getpid())
memory_info = process.memory_info()
rss_gb = memory_info.rss / (1024 ** 3)  # Convert from bytes to GB
vms_gb = memory_info.vms / (1024 ** 3)  # Convert from bytes to GB

print(f"CPU Memory Usage:")
print(f"RSS: {rss_gb:.2f} GB")
print(f"VMS: {vms_gb:.2f} GB")
print()

# GPU Memory Usage with GPUtil
import GPUtil

print("GPU Memory Usage (GPUtil):")
gpus = GPUtil.getGPUs()

for gpu in gpus:
    memoryTotal_gb = gpu.memoryTotal / 1024  # Convert from MB to GB
    memoryUsed_gb = gpu.memoryUsed / 1024  # Convert from MB to GB
    memoryFree_gb = gpu.memoryFree / 1024  # Convert from MB to GB

    print(f"GPU ID: {gpu.id}")
    print(f"Name: {gpu.name}")
    print(f"Total Memory: {memoryTotal_gb:.2f} GB")
    print(f"Used Memory: {memoryUsed_gb:.2f} GB")
    print(f"Free Memory: {memoryFree_gb:.2f} GB")
    print()

# GPU Memory Usage with pynvml
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

print("GPU Memory Usage (pynvml):")
nvmlInit()

# Get the number of GPUs
device_count = len(GPUtil.getGPUs())  # Correctly get the number of GPUs

for i in range(device_count):
    handle = nvmlDeviceGetHandleByIndex(i)
    info = nvmlDeviceGetMemoryInfo(handle)

    total_gb = info.total / (1024 ** 3)  # Convert from bytes to GB
    used_gb = info.used / (1024 ** 3)  # Convert from bytes to GB
    free_gb = info.free / (1024 ** 3)  # Convert from bytes to GB

    print(f"GPU ID: {i}")
    print(f"Total Memory: {total_gb:.2f} GB")
    print(f"Used Memory: {used_gb:.2f} GB")
    print(f"Free Memory: {free_gb:.2f} GB")
    print()

nvmlShutdown()

sys.exit(0)
