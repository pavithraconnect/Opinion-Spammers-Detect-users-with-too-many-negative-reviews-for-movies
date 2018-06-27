import org.apache.spark.ml.{Pipeline}
import org.apache.spark.ml.classification.{RandomForestClassifier,LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer ,StopWordsRemover}
import org.apache.spark.sql.functions._

object opinionSpam {
  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      println("5 parameters are required")
    }

    val sc = new SparkContext(new SparkConf().setAppName("spamDetect"))

    val spark = SparkSession
      .builder()
      .appName("SpamDetect")
      .getOrCreate()

    import spark.implicits._

    //Importing preprocesed dataset into a DataFrame
    val df1=spark.read.option("header","true").option("inferSchema","true").csv(args(0))

    //Splitting the helpfulness column in two columns, one for the numerator and one for the denominator
    val datadf1 = df1.withColumn("productId", $"productId").
      withColumn("userId", $"userId").
      withColumn("profileName", $"profileName").
      withColumn("helpfulness", $"helpfulness").
      withColumn("help_num", split($"helpfulness", "/").getItem(0)).
      withColumn("help_denom", split($"helpfulness", "/").getItem(1)).
      drop("helpfulness").
      withColumn("review_score", $"review_score").
      withColumn("review_time", $"review_time").
      withColumn("review_summary", $"review_summary").
      withColumn("text", $"text")

    //Removing rows with null values
    val data2 = datadf1.select("productId", "userId", "profileName", "help_num", "help_denom", "review_score", "review_time", "review_summary", "text")
    data2.na.drop().createOrReplaceTempView("reviews")

    val actualreviews1 = spark.sql("SELECT * FROM reviews")

    //Loading the AFINN file which conatins a sentiment score for individual words
    val sentiscore1 = sc.textFile(args(1)).map(x => x.split("\t")).map(x => (x(0).toString, x(1).toInt))

    //create a map of the word and the sentiment score and broadcast it to all nodes
    val sentimapfn1 = sentiscore1.collectAsMap.toMap
    val broadc = sc.broadcast(sentimapfn1)

    //Generate the sentiment score for each review
    val sentirevscore = actualreviews1.map(text => {
      val reviewWordsSentiment = text(8).toString.split(" ").map(word => {
        val senti: Int = broadc.value.getOrElse(word.toLowerCase(), 0)
        senti;
      });
      val reviewSentiment = reviewWordsSentiment.sum
      (text(0).toString, text(1).toString, text(2).toString, text(3).toString, text(4).toString, text(5).toString, text(6).toString, text(7).toString, text(7).toString, reviewSentiment)

    })

    sentirevscore.createOrReplaceTempView("review2")


    val sentisql = spark.sql("select _1 AS productID ,_2 AS userID ,_3 AS profileName, _4 AS Helpfulnr,_5 AS Helpfuldr, CAST(_6 AS INT) AS Score, CAST(_7 AS DOUBLE) AS time,_8 AS Summary,_9 AS Text,CAST(_10 AS INT) AS SentiScore from review2")


    sentisql.createOrReplaceTempView("ReviewsWithSentiment")

    //Average sentiment score of each user
    val xyz = spark.sql("select userID, avg(SentiScore) from ReviewsWithSentiment group by userID having avg(SentiScore)>20")

    //Compute Overall average sentiment score
    var avg_senti_score = spark.sql("select avg(SentiScore) as score from ReviewsWithSentiment ")

    //Label the reviews based on the threshold sentiment score which is calculated above
    import org.apache.spark.sql.functions._
    val BinarySummary = sentisql.withColumn("SentiScore", when($"SentiScore" >= -4.0, 1).otherwise(0))
    BinarySummary.show()
    BinarySummary.createOrReplaceTempView("labelled_data")

    data2.createOrReplaceTempView("data3")


    val mldatabase = spark.sql("SELECT productID,userID,Text,SentiScore as label FROM labelled_data")

    //Create train and test dataset
    val Array(training, test) = mldatabase.randomSplit(Array(0.8, 0.2), seed = 12345)

    //Create a pipeline
    //Extract words from the reviews
    //Remove Stop words
    //Extract features from the words
    //Create Machine Learning models and feed the data
    val tokenizer = new Tokenizer().setInputCol("Text").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(false)
    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
    val lr = new LogisticRegression().setRegParam(0.01).setThreshold(0.5)
    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, lr))

    //Training the Logistic Regression model
    val model = pipeline.fit(training)

    //Predicting for the unseen reviews
    var predictions = model.transform(test)

    //Evaluate the models
    val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
    val Area_LR = evaluator.evaluate(predictions)
    var modelstats="Area under the ROC curve Logistic Regression= " + evaluator.evaluate(predictions)

    //Constructing RandomForest Model
    val rf = new RandomForestClassifier()
      .setNumTrees(100)
      .setFeatureSubsetStrategy("auto")
    val pipeline_rf = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, rf))

    val model_rf = pipeline_rf.fit(training)

    val predictions_rf = model_rf.transform(test)

    //Evaluating RandomForest classifier
    val evaluator1 = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
     modelstats+="\nArea under the ROC curve Random Forest= " + evaluator1.evaluate(predictions_rf)
    val Area_RF = evaluator1.evaluate(predictions_rf)
    val paramGrid = new ParamGridBuilder().
      addGrid(lr.regParam, Array(0.4, 0.1, 0.2)).
      addGrid(lr.threshold, Array(0.5, 0.6, 0.7)).
      build()

    //5 fold cross validation
    val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5)


    val cvModel = cv.fit(training)

    modelstats+="\n\nArea under the ROC curve for non-tuned model = " + evaluator.evaluate(predictions_rf)
    modelstats+="\nArea under the ROC curve for fitted model = " + evaluator.evaluate(cvModel.transform(test))
    modelstats+="\nImprovement = " + "%.2f".format((evaluator.evaluate(cvModel.transform(test)) - evaluator.evaluate(predictions_rf)) * 100 / evaluator.evaluate(predictions_rf)) + "%"

    //Saving the results in files
    sc.parallelize(List(modelstats)).saveAsTextFile(args(4))

    predictions.rdd.saveAsTextFile(args(2))
    predictions_rf.rdd.saveAsTextFile(args(3))

  }
}