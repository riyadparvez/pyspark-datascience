FROM jupyter/pyspark-notebook

RUN conda install --quiet -y nltk jupyterthemes
RUN jt -t chesterish
