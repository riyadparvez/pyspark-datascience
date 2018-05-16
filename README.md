# pyspark-notebooks
PySpark Jupyter notebooks

Most of the notebooks are from Kaggle competitions or datasets from University of California at Irvine [Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php)

## Installation
We provide a [pre-built docker image](https://hub.docker.com/r/riyadparvez/pyspark-notebooks/) for easy experimentation. The docker image is based on offical jupyter [pyspark-notebook](https://hub.docker.com/r/jupyter/pyspark-notebook/) image. Some additional packages have been installed.

To pull the image:
`docker pull riyadparvez/pyspark-notebooks`

To run a container:
`docker run --rm -p 8888:8888 -p 8080:8080 -p 4040:4040 -v /path/to/pyspark-notebooks:/home/jovyan/work --name pyspark-notebook riyadparvez/pyspark-notebooks start-notebook.sh --NotebookApp.token=''`

Please see the documentation of official jupyter docker image for more usage.

# Notebooks
Most of the notebooks are WIP.
Complete notebooks are:

* [Titanic](https://nbviewer.jupyter.org/github/riyadparvez/pyspark-notebooks/blob/master/titanic/spark.ipynb)
* [Wine Quality](https://nbviewer.jupyter.org/github/riyadparvez/pyspark-notebooks/blob/master/wine-quality/spark.ipynb)
