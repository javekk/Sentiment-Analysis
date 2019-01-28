import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.pretrained.pipelines.en.BasicPipeline
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}

object SentimentTraining  {



  //def main(args: Array[String]): Unit = {
  def trainingMe(dataframeToTrain:DataFrame): DataFrame = {

    val spark: SparkSession = SparkSession
      .builder()
      .appName("test")
      .master("local[*]")
      .config("spark.driver.memory", "4G")
      .config("spark.kryoserializer.buffer.max", "200M")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val training = spark.read
      .option("mode", "DROPMALFORMED")
      .csv("trainNLP.csv")
      .withColumnRenamed("_c0", "train_text")
      .withColumnRenamed("_c1","train_sentiment")


    training.show()
    /*
    val testing = Array(
      "I don't recommend this movie, it's horrible",
      "Dont waste your time!!!"
    )
    */
    val testing = dataframeToTrain
    testing.show()

    val document = new DocumentAssembler()
      .setInputCol("train_text")
      .setOutputCol("document")

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols("token")
      .setOutputCol("normal")

    val vivekn = new ViveknSentimentApproach()
      .setInputCols("document", "normal")
      .setOutputCol("result_sentiment")
      .setSentimentCol("train_sentiment")

    val finisher = new Finisher()
      .setInputCols("result_sentiment")
      .setOutputCols("final_sentiment")

    val pipeline = new Pipeline().setStages(Array(document, token, normalizer, vivekn, finisher))

    val sparkPipeline = pipeline.fit(training)

    val lightPipeline = new LightPipeline(sparkPipeline)

    /*
    Benchmark.time("Light pipeline quick annotation") {
      lightPipeline.annotate(testing)
    }
    */




    Benchmark.time("Spark pipeline, this may be too much for just two rows!") {
      println("Updating DocumentAssembler input column")
      document.setInputCol("text")
      var returnDF = sparkPipeline.transform(testing).toDF("text","final_sentiment")
      returnDF.show()
      return returnDF
    }

  }
}
