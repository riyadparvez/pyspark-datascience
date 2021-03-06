from functools import reduce
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.sql.functions import col, countDistinct, format_number, lit, mean, stddev_pop, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline

import math
import numpy as np
import pandas as pd
import scipy.stats

import collections

class FrozenDict(collections.Mapping):
    """Don't forget the docstrings!!"""

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.items()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            self._hash = 0
            for pair in self.items():
                self._hash ^= hash(pair)
        return self._hash

# Returns (null column, fraction of missing values)
def findMissingValuesCols(df):
    n = df.count()
    nullCols = []
    for column in df.columns:
        c = df.filter(col(column).isNull()).count()
        if c > 0:
            nullCols.append((column, c/float(n)))
    return nullCols

def calcEntropy(df, *columns):
    n = df.count()
    entropies = {}
    for column in columns:
        aggr = df.groupby(column).count()
        rows = aggr.select((col('count') / n).alias('prob')).collect()
        probs = [row[0] for row in rows]
        entropies[column] = scipy.stats.entropy(probs)
    return entropies

def calcNormalizedEntropy(df, *columns):
    n = df.count()
    entropies = calcEntropy(df, *columns)
    normalizedEntropies = {}
    for column in columns:
        distinct = df.agg(countDistinct(column)).collect()[0][0]
        entropy = entropies[column]
        normalizedEntropy = entropy / math.log(distinct)
        normalizedEntropies[column] = normalizedEntropy
    return normalizedEntropies

def calcIndividualAndJointPorbablities(df, *columns):
    n = df.count()
    individualProbs = {}
    # Key: column, value: list of distinct values
    columnDistinctVals = {column: [] for column in columns}
    for column in columns:
        aggr = df.groupby(column).count()
        rows = aggr.withColumn('prob', (col('count') / n)).collect()
        for row in rows:
            val = row[column]
            key = (column, val)
            individualProbs[key] = row['prob']
            columnDistinctVals[column].append(val)

    aggr = df.groupby(*columns).count()
    rows = aggr.withColumn('prob', (col('count') / n)).collect()
    jointProbs = {}
    for row in rows:
        vals = tuple([(column, row[column]) for column in columns])
        prob = row['prob']
        jointProbs[vals] = prob
    return individualProbs, jointProbs

# High mutual information indicates a large reduction in uncertainty;
# low mutual information indicates a small reduction;
# and zero mutual information between two random variables means the
# variables are independent.
def calcMutualInformation(df, *columns) -> float:
    individualProbs, jointProbs = calcIndividualAndJointPorbablities(df, *columns)
    mutualInformation = 0
    for k, v in jointProbs.items():
        jointProb = v
        indProbs = [individualProbs[ind] for ind in k]
        g = jointProb * math.log(jointProb / reduce((lambda x, y: x * y), indProbs))
        mutualInformation += g

    return mutualInformation

def calcPointwiseMutualInformation(df, *columns):
    individualProbs, jointProbs = calcIndividualAndJointPorbablities(df, *columns)
    pmi = {}
    for k, v in jointProbs.items():
        jointProb = v
        indProbs = [individualProbs[ind] for ind in k]
        g = math.log(jointProb / reduce((lambda x, y: x * y), indProbs))
        pmi[k] = g
    return pmi

# no co-occurrences, logp(x,y)→−∞, so nmpi is -1,
# co-occurrences at random, logp(x,y)=log[p(x)p(y)], so nmpi is 0,
# complete co-occurrences, logp(x,y)=logp(x)=logp(y), so nmpi is 1.
def calcNormalizedPointwiseMutualInformation(df, *columns):
    individualProbs, jointProbs = calcIndividualAndJointPorbablities(df, *columns)
    npmi = {}
    for k, v in jointProbs.items():
        jointProb = v
        indProbs = [individualProbs[ind] for ind in k]
        pmi = math.log(jointProb / reduce((lambda x, y: x * y), indProbs))
        npmi[k] = pmi / -math.log(jointProb)
    return npmi

def calcNormalizedPointwiseMutualInformationPandas(df, *columns):
    npmi = calcNormalizedPointwiseMutualInformation(df, *columns)
    dfCols = columns + ['Normalized PMI']
    df_ = pd.DataFrame(columns=columns)
    for k, v in nmpi:
        npmi = v
        l = [*list(k.values()), npmi]
        print(l)
        # for  in k.values():

def calcNormalizedMutualInformation(df, col1: str, col2: str) -> float:
    entropies = calcEntropy(df, col1, col2)
    return (2* calcMutualInformation(df, col1, col2) / reduce(lambda x, y: x+y, entropies.values()))


def calcZScore(df, column: str, outputCol='zScore'):
    avg = df.agg(mean(column)).head()[0]
    stddev = df.agg(stddev_pop(column)).head()[0]

    def z_score(val):
        return (val-avg)/stddev

    df = df.withColumn(outputCol, udf(z_score, DoubleType())(df[column]))
    return df

def stratifiedSampling(df, key: str, fraction: float, seed=42):
    fractions = df.select(key).distinct().withColumn("fraction", lit(fraction)).rdd.collectAsMap()
    first = df.sampleBy(key, fractions, seed)
    second = df.subtract(first)
    return first, second

def crosstabPercentage(df, col1, col2):
    df2 = df.groupby(col1, col2).count().alias("df2")
    df3 = df.groupby(col1).count().alias("df3")
    df4 = df2.join(df3, df2[col1] == df3[col1], 'inner')\
             .select('df2.*', (col('df2.count') / col('df3.count')*100).alias('percentage'))\
             .orderBy(col1, col2)
    return df4

def autoIndexer(df, labelCol, maxDistinct=None, outputCol='assembled'):
    stringTypes = [dtype[0] for dtype in df.dtypes if dtype[1] == 'string']
    stringTypes.remove(labelCol)

    if maxDistinct:
        indexers = []
        for column in stringTypes:
            distinctCount = df.select(column).distinct().count()
            if distinctCount < maxDistinct:
                indexedCol = 'indexed' + column
                indexedCols.append(indexedCol)
                indexer = StringIndexer(inputCol=column, outputCol=indexedCol)
                indexers.append(indexer)
    else:
        indexers = [StringIndexer(inputCol=column, outputCol='indexed'+column) for column in stringTypes]

    indexedTypes = ['indexed'+column for column in stringTypes]
    oheTypes = [indexedType+'oneHotEncoded' for indexedType in indexedTypes]
    ohe = OneHotEncoderEstimator(inputCols=indexedTypes, outputCols=oheTypes)
    assembler = VectorAssembler(inputCols=oheTypes, outputCol=outputCol)
    pipeline = Pipeline(stages=[*indexers, ohe, assembler])
    indexed = pipeline.fit(df).transform(df)
    return stringTypes, oheTypes, indexed

def attributeValueFrequency(df, categoricalCols, outputCol='avfScore'):
    for column in categoricalCols:
        grouped = df.groupby(column).count().select(column, 'count')
        grouped = grouped.withColumnRenamed('count', column+'Count').alias('grouped')
        c = 'grouped.'+column+'Count'
        df = df.alias('df').join(grouped, df[column] == grouped[column], 'inner')\
                    .select('df.*', c)
    countCols = [column+'Count' for column in categoricalCols]
    colCols = [col(column) for column in countCols]
    expression = reduce(lambda x, y: x + y, colCols)

    df = df.withColumn(outputCol, expression).drop(*countCols)
    return df

def dictToPandasDF(dictionary, *columns):
    return pd.DataFrame(list(dictionary.items()), columns=[*columns])

def toPandasDF(dictionary, targetCol: str, *columns):
    dfCols =  [*columns, targetCol]
    rows = []
    for k, val in dictionary.items():
        d = {key: val for key, val in k}
        d[targetCol] = val
        rows.append(d)
    df_ = pd.DataFrame(rows, columns=dfCols).sort_values(by=list(columns))
    return df_
