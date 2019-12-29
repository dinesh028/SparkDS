package indore.dinesh.sachdev.uber

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.clustering._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._

object UberClusteringDriver {

  def main(args: Array[String]): Unit = {
    val schema = StructType(Array(
      StructField("dt", TimestampType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("base", StringType, true)))

    val spark: SparkSession = SparkSession.builder().appName("uber").getOrCreate()

    import spark.implicits._

    // path to dataset file
    var file: String = "/data/uber.csv"

    //Read the data set
    val df: Dataset[Uber] = spark.read
      .option("inferSchema", "false")
      .schema(schema)
      .csv(file).as[Uber]

    //Print Schema and first 5 rows
    df.printSchema
    df.show(5)

    //Define Features Array
    //a VectorAssembler transformer is used to return a new DataFrame with the input columns lat, lon in a vector features column
    // input column names
    val featureCols = Array("lat", "lon")

    // create transformer
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    // transform method adds features column
    val df2 = assembler.transform(df)
    // cache transformed DataFrame
    df2.cache
    df2.show(5)
    df2.printSchema()

    // create the estimator
    val kmeans: KMeans = new KMeans()
      .setK(20)
      .setFeaturesCol("features")
      .setPredictionCol("cid")
      .setSeed(1L)

    // use the estimator to fit (train) a KMeans model
    val model: KMeansModel = kmeans.fit(df2)

    // print out the cluster center latitude and longitude
    println("Final Centers: ")
    val centers = model.clusterCenters
    centers.foreach(println)

    // get the KMeansModelSummary from the KMeansModel
    val summary: KMeansSummary = model.summary
    // get the cluster centers in a dataframe column from the summary
    val clusters: DataFrame = summary.predictions
    // register the DataFrame as a temporary table
    clusters.createOrReplaceTempView("uber")
    clusters.show(5)

    //Which clusters had the highest number of pickups?
    clusters.groupBy("cid").count().orderBy(desc("count")).show(21)

    //Which hours of the day had the highest number of pickups?
    spark.sql("SELECT hour(uber.dt) as hr,count(cid) as ct FROM uber group By  hour(uber.dt) order by hour(uber.dt) ").show(24)

    //Which hours of the day and which cluster had the highest number of pickups?
    clusters.select(hour($"dt").alias("hour"), $"cid")
      .groupBy("hour", "cid").agg(count("cid")
        .alias("count")).orderBy(desc("count"), $"hour").show(5)

    //Which clusters had the highest number of pickups during morning rush hour?
    spark.sql("SELECT hour(uber.dt) as hr, cid, count(cid) as ct FROM uber WHERE cid IN (0,1,4,9, 10,11,16,15) and hour(uber.dt) IN (6,7,8,9) group By hour(uber.dt), cid order by hr, cid").show

    //Which clusters had the highest number of pickups during evening rush hour?
    spark.sql("SELECT hour(uber.dt) as hr, cid, count(cid) as ct FROM uber WHERE cid IN (0,1,4,9, 10,11,16,15) and hour(uber.dt) IN (16,17,18,19) group By hour(uber.dt), cid order by hr, cid").show
    
    //Save model
    model.write.overwrite().save("/data/uber")
    
    /*Load Model 
     * val sameModel = KMeansModel.load(savedirectory)*/
  }
}