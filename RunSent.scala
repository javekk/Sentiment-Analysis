
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
		 * Let's start with clean the record,
		 * We want to remove any tweet that doesn’t contain “happy” or “sad”. Why? 'coz
		 * we can easily make the assumption that if a tweet contains happy it's an happy
		 * tweet, and the same for sad.
		 * What about the "I am not happy" case???? --> Not handle
		 */

    val tweetDF = spark.read.option("mode", "DROPMALFORMED").csv("/home/jave/Documents/Madrid_eit_uni/mini_project-2/tweet_text_02_1M.csv")
    tweetDF.show()

    var text = tweetDF.select("_c2").withColumnRenamed("_c2", "text")
    text.show()


    var messages = text.select("text")
    println("Total messages: " + messages.count())


    /*
		 * Keep only the tweet that contains happy or sad, in order to be able to label them
		 */
    var happyMessages = messages.filter(messages("text").contains("happy"))
    val countHappy = happyMessages.count()
    println("Number of happy messages: " +  countHappy)

    var unhappyMessages = messages.filter(messages("text").contains(" sad"))
    val countUnhappy = unhappyMessages.count()
    println("Unhappy Messages: " + countUnhappy)

    val smallest = Math.min(countHappy, countUnhappy).toInt

    /*
		 * We keep an equal number sad and happy tweets in order prevent bias in the model
		 * then we create a new view using an an equal number of happy and sad tweets
		 */
    var tweets = happyMessages.limit(smallest).union(unhappyMessages.limit(smallest))
    tweets.show()


    /*
		 * @LABELINGOFDATA
		 * We remove the word happy and sad (or similar) from the tweet in order to
		 * infer the happiness or the sadness only via the other words, and also
		 * we label each tweet as 1 if happy or 0 if sad
		 */
    val messagesRDD = tweets.rdd
    val goodBadRecords = messagesRDD.map(
      row =>{
        Try{
          val msg = row(0).toString.toLowerCase()
          var isHappy:Int = 0
          if(msg.contains(" sad")){
            isHappy = 0
          }else if(msg.contains("happy")){
            isHappy = 1
          }
          var msgSanitized = msg.replaceAll("happy", "")
          msgSanitized = msgSanitized.replaceAll("sad","")
          //Return a tuple
          (isHappy, msgSanitized.split(" ").toSeq)
        }
      }
    )

    //We use this syntax to filter out exceptions
    val exceptions = goodBadRecords.filter(_.isFailure)
    println("total records with exceptions: " + exceptions.count())
    exceptions.take(10).foreach(x => println(x.failed))
    var labeledTweets = goodBadRecords.filter((_.isSuccess)).map(_.get)
    println("total records with successes: " + labeledTweets.count())

    labeledTweets.take(10).foreach(x => println(x))


    /*
		 * @DATATRANSFORMATION
		 * We use a Gradient Boosting algorithm that excepts an array of fixed lenght of number
		 * we hash each word into an fixed-lenght array
		 * we get an array that represents the count of ecah word in the tweet
		 * we use Bag-Of-Words
		 * for implement bow we use hashingTF
		 * we use an array of 3000 that seems it is enough
		 * since 3000 < #ofwords it is possible that two or more words
		 */
    val hashingTF = new HashingTF(3000)

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
    boostingStrategy.setNumIterations(30) //number of passes over our training data
    boostingStrategy.treeStrategy.setNumClasses(2) //We have two output classes: happy and sad
    boostingStrategy.treeStrategy.setMaxDepth(5)

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