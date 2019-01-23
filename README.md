# Sentiment-Analysis

In this project it is implement a model for a simple sentiment analysis, using Spark, HDFS and Scala/Java


This analysis is structured in a sequence of steps, those of which could be procedures for transforming data or specic algorithms to analyse the data itself.
First of all, a sentiment analysis is the process of determining the attitude (positive or negative) of the writer with respect to a particular topic. In particular, in this case the model produced focuses on the classication of each distinct tweet over a polarity dened as **happy** or **sad**.


## Technologies and Algorithms 

* **[Apache Spark](http://spark.apache.org/)**: The project has being developed on top of an open-source cluster-computing framework, Apache Spark, which is an unied analytic engine for large-scale data processing. 
* **[Scala](https://www.scala-lang.org/download/)**: Easier to use than Java.
* **[Java](https://www.java.com/en/)**: Maybe is not the best language to use in this case, but it is the most popular.
* **RDD, Dataset, Dataframe**: The main abstraction Spark provides is a Resilient Distributed Dataset which is a collection of elements partitioned across the nodes of the cluster that can be operated on in parallel. A Dataset is a distributed colleciton of data which benets of the optimised execution engine provided by SparkSQL. A DataFrame is a Dataset organized into named columns, which is conceptually equivalent to a table in a relational database.
* **Gradient Boosting**: In order to build a predictive model for this project, it was used a machine learning boosting technique called, *Gradient Boosting*. 
* **Bag of Words - HashingTF**: The Bag-of-Words model (BoW) is a way of representing text data in Natural Language Processing (NPL) and extracting features from text to use in modeling.
* **[HDFS](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html)**: Spark is an optimal choice for big data processing, although it does not come with its own le management system, so we considered Hadoop Distributed File System(HDFS).
* **[D3.js](https://d3js.org/)**: Finally, regarding the visualisation of the results coming from the analysis, it was chosen a JavaScript library, D3.js, which focused in manipulating documents based on data. In the file **HTML** there are some examples used to several database.

## Implementation
Regarding the implementation, we had to prepare the data before being able to process it, therefore we started by creating a builder, which will serve for the creation of the **SparkSession**, the entry point to program in Spark using the **Dataset** and **Dataframe** API. Created the Data Set, which contains all the tweets, we mapped it in two DataFrames, **happytweet** and **sadtweet** by using the *filter()* function. We created a general *Dataframe* with the same number of happy and sad tweets. Finally, we mapped the result from the **Dataframe** into a **RDD**(**JavaRDD**). This rdd is composed by a **tuple** (**Tuple2**) of integer (1 or 0) and an array of **Strings**(tweets' words). We used the **HashingTF** class on this rdd to have a new rdd of length array (3000), in order to give it as an input to the **Gradient Boosting** algorithm (with the help of the LabeledPoint class). Before giving this input, we divided it randomly into two, a **training set** (70%) and a **validation set** (30%). We set the GB paramenters (**BoostingStrategy**, **GradientBoostedTreesModel**). With the latter class, we created an object, which will helps us apply the boosting algorithm. Finally, the results of this algorithm are saved on **HDFS** through the **RDD**(**JavaRDD**).
