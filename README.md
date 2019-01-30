# Sentiment-Analysis

In this project it is implement a model for a simple **sentiment analysis** on **tweets**, using **Spark**, **NLP**, **HDFS** and **Scala**, and there is also a simpler version in **Java** without the NLP part.

This analysis is structured in a sequence of steps and in which it is implemented a process for determining the attitude (**positive** or **negative**) of the writer with respect to a particular topic. In particular, in this case the model produced focuses on the classification of each distinct tweet over a polarity as **happy** or **sad**.

## Technologies and Algorithms 

- **[Apache Spark](http://spark.apache.org/)**: The project has being developed on top of an open-source cluster-computing framework, Apache Spark, which is an unied analytic engine for large-scale data processing. 
- **[Scala](https://www.scala-lang.org/download/)**: Easier to use than Java.
- **[Java](https://www.java.com/en/)**: Maybe is not the best language to use in this case, but it is the most popular.
- [**John Snow - Spark-nlp**](https://github.com/JohnSnowLabs/spark-nlp):  **Natural language processing** libraries for Apache Spark. In this project it is used a simple pipeline, (pre-trained with a labeled dataset *trainNLP.csv*), for label our training set with a the polarity **negative** associated with **sad** and **positive** associated with **happy**.
- **RDD, Dataset, Dataframe**: The main abstraction Spark provides is a Resilient Distributed Dataset which is a collection of elements partitioned across the nodes of the cluster that can be operated on in parallel. A Dataset is a distributed collection of data which benefitts of the optimized execution engine provided by SparkSQL. A DataFrame is a Dataset organized into named columns, which is conceptually equivalent to a table in a relational database.
- **Gradient Boosting**: In order to build a predictive model for this project, it was used a machine learning boosting technique called, *Gradient Boosting*. 
- **Bag of Words - HashingTF**: The Bag-of-Words model (BoW) is a way of representing text data in Natural Language Processing (NPL) and extracting features from text to use in modeling.
- **[HDFS](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html)**: Spark is an optimal choice for big data processing, although it does not come with its own le management system, so we considered Hadoop Distributed File System(HDFS).
- **[D3.js](https://d3js.org/)**: Finally, regarding the visualisation of the results coming from the analysis, it was chosen a JavaScript library, D3.js, which focused in manipulating documents based on data. In the file **HTML** there are some examples used to several database.

## Implementation

##### From raw data to Dataset

In this code we use a big dataset with a lot of columns, but we care only about the text, so we select only that column. Of course this part of code is very different for every dataset we use.

##### Natural Language Process Labeling 

We want to **test** and use the **gradient boosting** model, and as it is a supervised model we need labeled records in order to train this model. There are a lot of different kind of approach, for example we could train the model directly with labeled dataset or use directly the NLP approach. But in this process we use want to use a **pre-trained pipeline** to label a tweet as positive or negative in order to build our training set, or rather, we simulate it :) . With a piece of code got from [here](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/src/english/TrainViveknSentiment.scala) and the dataset got from [here](http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) (well-prepared -> trainNLP.csv) we train the pipeline on every time we run the code with that dataset. (as already said there are better approaches but we do not take them into account because meh). With this we are able than to label the real tweet dataset with 1 happy or 0 sad.

##### Data transformation

We use gradient boosting algorithm and it accepts a fixed-length array of numbers (not strings). So in order to transform a tweet to a fixed-length-numeric array we use a technique called **Bag of Word (BOW)** i.e. we hash the words into that array and in the end we get an array which represents the count of each word in the tweet. The implementation of this is done by function called **HashingTF**. The length of this array is something which is possible to **boost**, 2000 seems a good value to start.

##### Build and evaluate the model

First we split the dataset into training and testing set and we finally apply Gradient Boosting, the parameters in which we can tweak to take into account are (1) **percentage of splitting** between the two sets, (2) **number of iterations** over the data and (3) **maximum deep** of each tree. After that we are done, and we can evaluate the model, comparing the predictions with the actual values, and than if it possible for example to plot this data in some graphic, using D3js for example (as I did).

### Java version

Regarding the implementation of the Java version there are some important differences, mainly due on the fact we do not use NLP here but a simpler approach. In this implementation we simply take only the tweets contain the word 'happy' or 'sad' and we label a tweet contains 'happy' as happy indeed, and vice-versa with sad, and we remove the others tweets :D .
