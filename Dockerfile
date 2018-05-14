FROM jupyter/pyspark-notebook

LABEL maintainer="riyad.parvez@gmail.com"

RUN conda install --quiet -y nltk jupyterthemes
RUN jt -t chesterish
EXPOSE 4040 8080
