package bigdata.sparky;

import static org.apache.spark.sql.functions.col;

import java.sql.Time;
import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.configuration.Strategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.mllib.tree.model.Predict;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;
import shapeless.labelled;


public class SentAnalysis {
	
	private static final String topic = "pizza";
	
	/*
	 * Some variables using for the evaluating of the model
	 */
	static long happyTotal = 0;
	static long sadTotal = 0;
	static long happyCorrect = 0;
	static long sadCorrect = 0;
	
	/*
	 * The Schema of tweets' text
	 * We are going to use only Text
	 * 
	 * 		tweetText(tweet_id,user_id,text)
	 * 
	 */
	static StructType schemaText = new StructType()
			.add("tweet_id", "long")
			.add("user_id", "long")
			.add("text", "String")
			.add("geo_lat", "long")
			.add("geo_long", "long")
			.add("place_full_name", "String")
			.add("place_id", "long");

	/*
	 * 
	 * 
	 * @author Raffaele Perini, Giovanni Rafaèl Vuolo
	 * 
	 * 
	 * 
	 */
	public static void main(String[] args) {

		// SparkSession
		Builder builder = new Builder().appName("SparkSQL Examples");
		builder.master("local");


		SparkSession spark = builder.getOrCreate();


		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);


		/*
		 * 	Regarding the tweet's texts all we need is tweet_id, user_id, text
		 *  So we do a select on the tweet table
		 *  then we print the schema	
		 */
		Dataset<Row> text = spark
				.read()
				.option("mode", "DROPMALFORMED")
				.schema(schemaText)
				.csv("/home/jave/Documents/UNIMASTER/bigData/data_02/tweet_text_02.csv");

		text = text.select(col("text"));

		text.printSchema();


		/*
		 * @FROMRAWDATATODATAFRAME
		 * Let's start with clean the record, 
		 * We want to remove any tweet that doesn’t contain “happy” or “sad”. Why? 'coz 
		 * we can easily make the assumption that if a tweet contains happy it's an happy 
		 * tweet, and the same for sad.
		 * What about the "I am not happy" case???? --> Not handle
		 */ 

		//text.select("text").createOrReplaceTempView("text");
		long analysedTweetsNumber = text.count();
		System.out.println("There are : " + analysedTweetsNumber + " tweets");

		/*
		 * Keep only the tweet that contains happy or sad
		 */
		Dataset<Row> happyTweet = text.filter(col("text").contains("happy"));
		long counterOfHappyTweets = happyTweet.count();
		happyTweet.show();
		Dataset<Row> sadTweet = text.filter(col("text").contains("sad"));
		long counterOfSadTweets = sadTweet.count();
		sadTweet.show();
		System.out.println("#happy: " + counterOfHappyTweets + "\n#sad: " + counterOfSadTweets);
		
		/*
		 * We keep an equal number sad and happy tweets in order prevent bias in the model
		 * then we create a new view using an an equal number of happy and sad tweets
		 */
		long minNumber = Math.min(counterOfHappyTweets, counterOfSadTweets);
		System.out.println(minNumber);
		Dataset<Row> tmp1 = happyTweet.limit((int) minNumber).union(sadTweet.limit((int) minNumber));	//LOOK FOR UNIONALL
		tmp1.show();

		/*
		 * @LABELINGOFDATA
		 * We remove the word happy and sad (or similar) from the tweet in order to 
		 * infer the happiness or the sadness only via the other words, and also 
		 * we label each tweet as 1 if happy or 0 if sad
		 */

		JavaRDD<Row> textRdd = tmp1.javaRDD();

		JavaRDD<Tuple2<Integer,String[]>> statusAndSplit = textRdd.map(
				(Function<Row,Tuple2<Integer,String[]>>)s -> {
					try {
						String txt = s.getString(0).toLowerCase();
						int isHappy = 0;
						if(txt.contains("happy")) {
							isHappy = 1;
						}else if(txt.contains("sad")) {
							isHappy = 0;
						}
						txt = txt.replaceAll("happy", "");
						txt = txt.replaceAll("sad", "");
						Tuple2<Integer,String[]> ret = new Tuple2<>(isHappy, txt.split(" "));
						return ret;

					}catch(Exception e){
						System.out.println(e);
					};
					return null;
				});

		long exceptions = textRdd.count() - statusAndSplit.count();
		System.out.println("Number of records now:" + statusAndSplit.count());
		System.out.println("Number of exceptions: " + exceptions);

		for(int i = 0; i < 5; i++) {
			System.out.println(statusAndSplit.take(5).get(i));
		}

		/*
		 * @DATATRANSFORMATION
		 * We use a Gradient Boosting algorithm that excepts an array of fixed lenght of number
		 * we hash each word into an fixed-lenght array
		 * we get an array that represents the count of ecah word in the tweet
		 * we use Bag-Of-Words 
		 * for implement bow we use hashingTF
		 * we use an array of 3000 that seems it is enough
		 * since 3000 < #ofwords it is possibile that two or more words 
		 */

		HashingTF hashingTF = new HashingTF(3000);

		JavaRDD<LabeledPoint> textRdd2 = 
				statusAndSplit
				.map((Function<Tuple2<Integer,String[]>,Tuple2<Integer,Vector>>) t -> {
					return new Tuple2<Integer,Vector>(
							t._1, 
							hashingTF.transform(Arrays.asList(t._2)));
				})
				.map((Function<Tuple2<Integer,Vector>,LabeledPoint>)t -> {
					return new LabeledPoint(t._1.doubleValue(), t._2);
				});

		for(int i = 0; i < 10; i++) {
			System.out.println(textRdd2.take(10).get(i));
		}
		
		/*
		 * @SPLITOFSETS
		 * 30% for the validation set
		 * 
		 */
		
		
		// Split the data into training and test sets (30% held out for testing)
		JavaRDD<LabeledPoint>[] splits = textRdd2.randomSplit(new double[] {0.7, 0.3});
		JavaRDD<LabeledPoint> trainingData = splits[0];
		JavaRDD<LabeledPoint> validationData = splits[1];
		
		
		/*
		 * @BUILDTHEMODEL
		 * 30 passes over the trainingset
		 * 2 classes, happy or sad
		 * 6 depth of each tree. The higher it is, the higher probability of overfitting
		 * the lower it is, the simpler the model is
		 */
		
		BoostingStrategy bs = BoostingStrategy.defaultParams("Classification");
		bs.setNumIterations(30);
		bs.treeStrategy().setNumClasses(2);
		bs.treeStrategy().setMaxDepth(10);
		
		GradientBoostedTreesModel model = GradientBoostedTrees.train(trainingData, bs);
		
		/*
		 * @EVALUATETHEMODEL
		 * 
		 * 
		 */
		
		JavaRDD<Tuple2<Double,Double>> labelAndPredictTrain = trainingData.map(
				x -> {
					double prediction = model.predict(x.features());
					return new Tuple2<Double,Double>(x.label(), prediction);
				});
				
		JavaRDD<Tuple2<Double,Double>> labelAndPredictValid = validationData.map(
				x -> {
					double prediction = model.predict(x.features());
					return new Tuple2<Double,Double>(x.label(), prediction);
				});
		
		
		List<Tuple2<Double,Double>> results = labelAndPredictTrain.collect();
	
		results.forEach(
				x -> {
					if(x._1 == 1) {happyTotal++;}
					else if(x._1 == 0) {sadTotal++;}
					if(x._1 == 1 && x._2 == 1) {happyCorrect++;}
					else if(x._1 == 0 && x._2 == 0) {sadCorrect++;}
				});
		
		System.out.println("\n\n=========" );
		System.out.println(" TOPIC ----------------> " + topic  );
		System.out.println(" #Analysed Tweets -----> " + analysedTweetsNumber + "\n");
		
		
		System.out.println("=====TRAIN=====" );
		System.out.println("Total happys+sad: " + (happyTotal + sadTotal));

		System.out.println("=HAPPY=: " );
		System.out.println("Total happys: " + happyTotal );
		System.out.println("Total correct happys: " + happyCorrect );
		System.out.println("Percentage of correct happys : " + new Double(happyCorrect)/new Double(happyTotal) );
		
		System.out.println("=SAD=: " );
		System.out.println("Total sads: " + sadTotal );
		System.out.println("Total correct sads: " + sadCorrect );
		System.out.println("Percentage of correct sads : " + new Double(sadCorrect)/ new Double(sadTotal) );
		
		System.out.println("=ERROR=: " );
		long tmpEr = (happyTotal + sadTotal) - (happyCorrect + sadCorrect);
		double error = new Double(tmpEr) / new Double(happyTotal + sadTotal);
		System.out.println("Total error: " + tmpEr );
		System.out.println("Total error percentage: " + error );
		
		
		
		results = labelAndPredictValid.collect();
		
		results.forEach(
				x -> {
					if(x._1 == 1) {happyTotal++;}
					else if(x._1 == 0) {sadTotal++;}
					if(x._1 == 1 && x._2 == 1) {happyCorrect++;}
					else if(x._1 == 0 && x._2 == 0) {sadCorrect++;}
				});
		
		System.out.println("\n\n=====VALID=====" );
		System.out.println("Total happys+sad: " + (happyTotal + sadTotal));

		System.out.println("=HAPPY=: " );
		System.out.println("Total happys: " + happyTotal );
		System.out.println("Total correct happys: " + happyCorrect );
		System.out.println("Percentage of correct happys : " + new Double(happyCorrect)/new Double(happyTotal) );
		
		System.out.println("=SAD=: " );
		System.out.println("Total sads: " + sadTotal );
		System.out.println("Total correct sads: " + sadCorrect );
		System.out.println("Percentage of correct sads : " + new Double(sadCorrect)/ new Double(sadTotal) );
		
		System.out.println("=ERROR=: " );
		tmpEr = (happyTotal + sadTotal) - (happyCorrect + sadCorrect);
		error = new Double(tmpEr) / new Double(happyTotal + sadTotal);
		System.out.println("Total error: " + tmpEr );
		System.out.println("Total error percentage: " + error );
		
		System.out.println(new Time(System.currentTimeMillis()).toGMTString().toString());
		
		labelAndPredictTrain.saveAsTextFile("hdfs://localhost:9000/bigdata/project/train.csv");
		labelAndPredictValid.saveAsTextFile("hdfs://localhost:9000/bigdata/project/valid.csv");
	}
}

