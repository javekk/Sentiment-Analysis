
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.sql.SparkSession

import scala.util.{Success, Try}



object RunSent{

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)


    val conf = new SparkConf().setAppName("mini-project2").setMaster("local")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder()
      .appName("mini-project2")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    /*
		 * @FROMRAWDATATODATAFRAME
		 * We use a Twitter dataset, but we care only about the tweet text
		 */

    val tweetDF = spark.read.option("mode", "DROPMALFORMED").csv("tweets.csv")
    tweetDF.show()

    var messages = text.select("text")
    println("Total messages: " + messages.count())

    
    /*
		 * @LABELINGOFDATA
		 * I use an NLP <see other class>
		 */

    val tmp= SentimentTraining.trainingMe(messages).rdd
    val records = tmp.map(
      row =>{
        Try{
          val sentiment= row(1).toString.toLowerCase()
          var msgSanitized = sentiment.replace("max", "max") // placeholder
          var isHappy = 0
          if(sentiment.contains("positive")){
            isHappy = 1
            msgSanitized = sentiment.replace("positive", "1")
          }else if(sentiment.contains("negative")){
            msgSanitized = sentiment.replace("negative", "0")
            isHappy = 0
          }
          //Return a tuple
          (isHappy, row(0).toString.split(" ").toSeq)
        }
      }
    )

    println("RECORDS")
    records.take(20).foreach(x => println(x+"e"))


    //We use this syntax to filter out exceptions
    val exceptions = records.filter(_.isFailure)
    println("total records with exceptions: " + exceptions.count())

    exceptions.take(20).foreach(x => println(x+"e"))

    var labeledTweets = records.filter((_.isSuccess)).map(_.get)
    println("total records with successes: " + labeledTweets.count())

    labeledTweets.take(20).foreach(x => println(x+"l"))


    /*
		 * @DATATRANSFORMATION
		 * We use a Gradient Boosting algorithm that excepts an array of fixed lenght of number
		 * we hash the words into an fixed-lenght array
		 * we get an array that represents the count of ecah word in the tweet
		 * we use Bag-Of-Words
		 * for implement bow we use hashingTF
		 * we use an array of 3000 that seems it is enough
		 * since 3000 < #ofwords it is possible that two or more words
		 */
    val hashingTF = new HashingTF(2000)

    //Map the input strings to a tuple of labeled point + input text
    val input_labeled = (labeledTweets.map(
      t => (t._1, hashingTF.transform(t._2)))
      .map(x => new LabeledPoint((x._1).toDouble, x._2)))

    input_labeled.take(10).foreach(println)

  
    /*
		 * @SPLITOFSETS
		 * 30% for the validation set
		 *
		 */
    val splits = input_labeled.randomSplit(Array(0.7, 0.3))
    val (trainingData, validationData) = (splits(0), splits(1))


    /*
		 * @BUILDTHEMODEL
		 * 30 passes over the trainingset
		 * 2 classes, happy or sad
		 * 6 depth of each tree. The higher it is, the higher probability of overfitting
		 * the lower it is, the simpler the model is
		 */
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(25) //number of passes over our training data
    boostingStrategy.treeStrategy.setNumClasses(2) //We have two output classes: happy and sad
    boostingStrategy.treeStrategy.setMaxDepth(7)

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    /*
		 * @EVALUATETHEMODEL
		 *
		 * and compute error
		 */
    var labelAndPredsTrain = trainingData.map { point =>
      val prediction = model.predict(point.features)
      Tuple2(point.label, prediction)
    }

    var labelAndPredsValid = validationData.map { point =>
      val prediction = model.predict(point.features)
      Tuple2(point.label, prediction)
    }

    //Since Spark has done the heavy lifting already, lets pull the results back to the driver machine.
    //Calling collect() will bring the results to a single machine (the driver) and will convert it to a Scala array.

    //Start with the Training Set
    val results = labelAndPredsTrain.collect()

    var happyTotal = 0
    var unhappyTotal = 0
    var happyCorrect = 0
    var unhappyCorrect = 0

    val resultsTr = labelAndPredsTrain.collect().foreach(
      r => {
        if (r._1 == 1) {
          happyTotal += 1
        } else if (r._1 == 0) {
          unhappyTotal += 1
        }
        if (r._1 == 1 && r._2 ==1) {
          happyCorrect += 1
        } else if (r._1 == 0 && r._2 == 0) {
          unhappyCorrect += 1
        }
      }
    )
    println("Training Set:")
    println("    unhappy messages " + unhappyTotal)
    println("    happy messages: " + happyTotal)
    println("    happy % correct: " + happyCorrect.toDouble/happyTotal)
    println("    unhappy % correct: " + unhappyCorrect.toDouble/unhappyTotal)

    val testErrTr = labelAndPredsTrain.filter(r => r._1 != r._2).count.toDouble / trainingData.count()
    println("    test Error :" + testErrTr)


    happyTotal = 0
    unhappyTotal = 0
    happyCorrect = 0
    unhappyCorrect = 0
    val resultsVa = labelAndPredsValid.collect().foreach(
      r => {
        if (r._1 == 1) {
          happyTotal += 1
        } else if (r._1 == 0) {
          unhappyTotal += 1
        }
        if (r._1 == 1 && r._2 ==1) {
          happyCorrect += 1
        } else if (r._1 == 0 && r._2 == 0) {
          unhappyCorrect += 1
        }
      }
    )
    println("Validation Set ")
    println("    unhappy messages" + unhappyTotal)
    println("    happy messages: " + happyTotal)
    println("    happy % correct: " + happyCorrect.toDouble/happyTotal)
    println("    unhappy % correct: " + unhappyCorrect.toDouble/unhappyTotal)

    val testErrVa = labelAndPredsValid.filter(r => r._1 != r._2).count.toDouble / validationData.count()
    println("    test error: " + testErrVa)




  }

}
