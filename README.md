# Sentiment-Analysis
Project it is implement a model for a simple sentiment analysis, using Spark, HDFS and Java


This analysis is structured in a sequence of steps, those of which could be procedures for transforming data or specic algorithms to analyse the data itself.
First of all, a sentiment analysis is the process of determining the attitude (positive or negative) of the writer with respect to a particular topic. In particular, in this case the model produced focuses on the classication of each distinct tweet over a polarity dened as **happy** or **sad**.


## Technologies and Algorithms 

* **Apache Spark**: The project has being developed on top of an open-source cluster-computing framework, Apache Spark, which is an unied analytic engine for large-scale data processing.
* **Java**: Maybe is not the best language to use in this case, but it is the most popular.
* **RDD, Dataset, Dataframe**: The main abstraction Spark provides is a Resilient Distributed Dataset which is a collection of elements partitioned across the nodes of the cluster that can be operated on in parallel. A Dataset is a distributed colleciton of data which benets of the optimised execution engine provided by SparkSQL. A DataFrame is a Dataset organized into named columns, which is conceptually equivalent to a table in a relational database.
* **Gradient Boosting**: In order to build a predictive model for this project, it was used a machine learning boosting technique called, *Gradient Boosting*. 
* **Bag of Words - HashingTF**: The Bag-of-Words model (BoW) is a way of representing text data in Natural Language Processing (NPL) and extracting features from text to use in modeling.
* **HDFS**: Spark is an optimal choice for big data processing, although it does not come with its own le management system, so we considered Hadoop Distributed File System(HDFS).
* (Extra)*D3.js*: Finally, regarding the visualisation of the results coming from the analysis, it was chosen a JavaScript library, D3.js, which focused in manipulating documents based on data. In the file **HTML** there are some examples used to several database.
