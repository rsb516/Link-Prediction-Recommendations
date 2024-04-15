import networkx as nx
import pandas as pd
import numpy as np
import arxiv

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec as n2v

queries = [
    'automl', 'machine learning', 'data', 'physics', 'mathematics', 'recommendation system', 'nlp', 'neural networks'
]

d = []
searches  = []
max_results = 100

# making request from the API

for query in queries:
    search  = arxiv.Search(
        query = query,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order = arxiv.SortOrder.Descending
    )
    searches.append(search)

# Making the search results into a dataframe
for search in searches:
    for res in search.results():
        data = {
            'title': res.title,
            'date': res.published,
            'article_id': res.entry_id,
            'url': res.pdf_url,
            'main_topic': res.primary_category,
            'all_topics': res.categories,
            'authors': res.authors

        }
        d.append(data)
        
d = pd.DataFrame(d)

print(d.head(5))
    
#print(searches)