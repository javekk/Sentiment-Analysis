name := "sentiment_analysis"

version := "0.1"

scalaVersion := "2.11.8"


val sparkVersion = "1.6.1"

libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "1.8.0"
libraryDependencies +=  "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies +=   "org.apache.spark" %% "spark-streaming" % sparkVersion
libraryDependencies +=  "org.apache.spark" %% "spark-streaming-twitter" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.2"
