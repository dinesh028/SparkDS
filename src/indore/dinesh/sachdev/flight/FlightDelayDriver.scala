package indore.dinesh.sachdev.flight

import org.apache.spark.sql.types._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.RandomForestClassificationModel

object FlightDelayDriver {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().enableHiveSupport().getOrCreate()

    /*Schema*/
    val schema = StructType(Array(
      StructField("_id", StringType, true),
      StructField("dofW", IntegerType, true),
      StructField("carrier", StringType, true),
      StructField("origin", StringType, true),
      StructField("dest", StringType, true),
      StructField("crsdephour", IntegerType, true),
      StructField("crsdeptime", DoubleType, true),
      StructField("depdelay", DoubleType, true),
      StructField("crsarrtime", DoubleType, true),
      StructField("arrdelay", DoubleType, true),
      StructField("crselapsedtime", DoubleType, true),
      StructField("dist", DoubleType, true)))

    var file = "D:/data/flights.json"

    import spark.implicits._
    val df = spark.read.format("json").option("inferSchema", "false").schema(schema).load(file).as[Flight]
    //df.createOrReplaceTempView("flights")

    /*Print the results*/
    df.show(3)

    val df1 = df.withColumn("orig_dest", concat($"origin", lit("_"), $"dest"))

    /*Filter Origination and destination having departure delay > 40 . And get the counts ordered descending */
    df1.select($"orig_dest", $"depdelay").filter($"depdelay" > 40).groupBy($"orig_dest").count.orderBy(desc("count")).show(5)

    /* perform summary statistics calculations on all numeric columns*/
    df.describe("dist", "depdelay", "arrdelay", "crselapsedtime").show

    /*A Spark Bucketizer is used to split the Dataset into delayed and not delayed flights with a delayed 0.0/1.0 column*/
    val delaybucketizer = new Bucketizer().setInputCol("depdelay").setOutputCol("delayed").setSplits(Array(0.0, 15.0, Double.PositiveInfinity))

    val df2 = delaybucketizer.transform(df1)
    df2.createOrReplaceTempView("flights")
    df2.groupBy("delayed").count.show

    /*Stratified Sampling*/
    /*Here, we’re keeping all instances of delayed, but down sampling the not delayed instances to 13%*/
    val fractions = Map(0.0 -> .13, 1.0 -> 1.0)
    val strain = df2.stat.sampleBy("delayed", fractions, 36L)

    strain.groupBy("delayed").count.show

    /*Then we split data in to training and test data as 0.7 and 0.3 percent respectively */
    val Array(trainingData, testData) = strain.randomSplit(Array(0.7, 0.3), 5043)

    /*====================================================================================*/
    /* Historical Data Analyzed- formed Training and Test Data set 												*/
    /*====================================================================================*/

    /*The code below sets up StringIndexers for all of the categorical columns*/

    val categoricalColumns = Array("carrier", "origin", "dest", "dofW", "orig_dest")
    /*Indices are fit to data set*/
    val stringIndexers = categoricalColumns.map { colName => new StringIndexer().setInputCol(colName).setOutputCol(colName + "Indexed").fit(strain) }

    // add a label column based on departure delay
    val labeler = new Bucketizer().setInputCol("depdelay").setOutputCol("label").setSplits(Array(0.0, 40.0, Double.PositiveInfinity))

    /*The VectorAssembler is used in the pipeline to combine a given list of columns into a single feature vector column.*/
    val featureCols = Array("carrierIndexed", "destIndexed",
      "originIndexed", "dofWIndexed", "orig_destIndexed",
      "crsdephour", "crsdeptime", "crsarrtime",
      "crselapsedtime", "dist")

    // combines a list of feature columns into a vector column
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    //StringIndexer, Bucketizer, VectorAssembler are transformers

    // Then we have estimator - Random Forest Classifier which will train on the vector of labels and features and return a (transformer) model
    val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")

    val steps = stringIndexers ++ Array(labeler, assembler, rf)

    // define the pipeling with all stages
    val pipeline = new Pipeline().setStages(steps)

    /*Train The Model*/
    //we use a ParamGridBuilder to construct the parameter grid for the model training.
    //We define an evaluator, which will evaluate the model by comparing the test label column with the test prediction column.
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxBins, Array(100, 200))
      .addGrid(rf.maxDepth, Array(2, 4, 10))
      .addGrid(rf.numTrees, Array(5, 20))
      .addGrid(rf.impurity, Array("entropy", "gini"))
      .build()

    val evaluator = new BinaryClassificationEvaluator()

    // Set up 3-fold cross validation with paramGrid
    val crossvalidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid).setNumFolds(3)

    // fit the training data set and return a model
    val pipelineModel = crossvalidator.fit(trainingData)

    /*The CrossValidator uses the ParamGridBuilder to iterate through the maxDepth, maxBins,
     *  and numbTrees parameters of the Random Forest Classifier and to evaluate the models,
     *   repeating 3 times per parameter value for reliable results.*/

    val featureImportances = pipelineModel
      .bestModel.asInstanceOf[PipelineModel]
      .stages(stringIndexers.size + 2)
      .asInstanceOf[RandomForestClassificationModel]
      .featureImportances

    assembler.getInputCols.zip(featureImportances.toArray).sortBy(-_._2).foreach { case (feat, imp) => println(s"feature: $feat, importance: $imp") }

    //best random forest model produced, using the cross-validation process
    val bestEstimatorParamMap = pipelineModel.getEstimatorParamMaps.zip(pipelineModel.avgMetrics).maxBy(_._2)._1
    println(s"Best params:\n$bestEstimatorParamMap")

    /*Predictions and Model Evaluation*/
    val predictions = pipelineModel.transform(testData)

    predictions.show

    /*The closer the value is to 1, the better its predictions are.*/
    val areaUnderROC = evaluator.evaluate(predictions)

    println(areaUnderROC)

    val lp = predictions.select("label", "prediction")
    val counttotal = predictions.count()
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(not($"label" === $"prediction")).count()
    val ratioWrong = wrong.toDouble / counttotal.toDouble
    val ratioCorrect = correct.toDouble / counttotal.toDouble

    val truep = lp.filter($"prediction" === 0.0)
      .filter($"label" === $"prediction").count() /
      counttotal.toDouble

    val truen = lp.filter($"prediction" === 1.0)
      .filter($"label" === $"prediction").count() /
      counttotal.toDouble

    val falsep = lp.filter($"prediction" === 0.0)
      .filter(not($"label" === $"prediction")).count() /
      counttotal.toDouble

    val falsen = lp.filter($"prediction" === 1.0)
      .filter(not($"label" === $"prediction")).count() /
      counttotal.toDouble

    val cor = truen + truep / falsep + falsen

    /*Save the model for later use*/
    pipelineModel.write.overwrite().save("D:/data/flightsmodel")
    
    /*To reload the model
    val sameModel = CrossValidatorModel.load("D:/data/flightsmodel")*/

    println("ratio correct", ratioCorrect)

    println("true positive", truep)

    println("false positive", falsep)

    println("true negative", truen)

    println("false negative", falsen)
    
    val precision = truep / (truep + falsep)
    val recall = truep / (truep + falsen)
    val fmeasure = 2 * precision * recall / (precision + recall)
    val accuracy = (truep + truen) / (truep + truen + falsep + falsen)

    println("precision ", precision)
    println("recall " + recall)
    println("f_measure " + fmeasure)
    println("accuracy " + accuracy)
  }
}
