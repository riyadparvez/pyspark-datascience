from pyspark.sql.functions import col
from scipy.stats import entropy

import math
import numpy as np
import pandas as pd

def findMissingValuesCols(df):
    numRows = df.count()
    nullCols = []
    for column in df.columns:
        c = df.filter(col(column).isNotNull()).count()
        if c != numRows:
            nullCols.append(c)
    return nullCols

def calcEntropy(df, *columns):
    n = df.count()
    entropies = {}
    for column in columns:
        aggr = df.groupby(column).count()
        rows = aggr.select((col('count') / n).alias('prob')).collect()
        probs = [row[0] for row in rows]
        entropies[column] = entropy(probs)
    return entropies

# High mutual information indicates a large reduction in uncertainty;
# low mutual information indicates a small reduction;
# and zero mutual information between two random variables means the
# variables are independent.
def calcMutualInformation(df, *columns):
    n = df.count()
    individualProbs = {}
    # Key: column, value: list of distinct values
    columnDistinctVals = {column: [] for column in columns}
    for column in columns:
        aggr = df.groupby(column).count()
        rows = aggr.withColumn('prob', (col('count') / n)).collect()
        for row in rows:
            val = row[column]
            individualProbs[(column, val, )] = row['prob']
            columnDistinctVals[column].append(val)

    aggr = df.groupby(*columns).count()
    rows = aggr.withColumn('prob', (col('count') / n)).collect()
    jointProbs = {}
    for row in rows:
        vals = tuple([(column, row[column]) for column in columns])
        prob = row['prob']
        jointProbs[vals] = prob

    mutualInformation = 0
    for k, v in jointProbs.items():
        jointProb = v
        indProbs = [individualProbs[ind] for ind in k]
        g = jointProb * math.log(jointProb / reduce((lambda x, y: x * y), indProbs))
        mutualInformation += g

    return mutualInformation

def calcNormalizedMutualInformation(df, col1, col2):
    entropies = calcEntropy(df, col1, col2)
    return (2* calcMutualInformation(df, col1, col2) / reduce(lambda x, y: x+y, entropies.values()))
