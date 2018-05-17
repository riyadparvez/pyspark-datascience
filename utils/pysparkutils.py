from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, lit
from scipy.stats import entropy

import math
import numpy as np
import pandas as pd

def findMissingValuesCols(df):
    nullCols = []
    for column in df.columns:
        c = df.filter(col(column).isNull()).count()
        if c > 0:
            nullCols.append(c)
    return nullCols

def autoIndexer(df, maxDistinct):
    stringTypes = [dtype[0] for dtype in df.dtypes if dtype[1] == 'string']
    indexed = df
    indexedCols = []
    for column in stringTypes:
        distinctCount = df.select(column).distinct().count()
        if distinctCount < maxDistinct:
            indexedCol = 'Indexed' + column
            indexedCols.append(indexedCol)
            indexer = StringIndexer(inputCol=column, outputCol=indexedCol)
            indexed = indexer.fit(indexed).transform(indexed)
            indexed = indexed.drop(column)
    return indexedCols, indexed

def calcEntropy(df, *columns):
    n = df.count()
    entropies = {}
    for column in columns:
        aggr = df.groupby(column).count()
        rows = aggr.select((col('count') / n).alias('prob')).collect()
        probs = [row[0] for row in rows]
        entropies[column] = entropy(probs)
    return entropies

def calcNormalizedEntropy(df, *columns):
    n = df.count()
    entropies = {}
    for column in columns:
        distinct = df.agg(countDistinct(column)).collect()[0][0]
        aggr = df.groupby(column).count()
        rows = aggr.select((col('count') / n).alias('prob')).collect()
        probs = [row[0] for row in rows]
        entropy = scipy.stats.entropy(probs)
        normalizedEntropy = entropy / math.log(distinct)
        entropies[column] = normalizedEntropy
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

def stratifiedSampling(df, key, fraction):
    fractions = df.select(key).distinct().withColumn("fraction", lit(fraction)).rdd.collectAsMap()
    first = df.sampleBy(key, fractions, 42)
    second = df.subtract(first)
    return first, second
