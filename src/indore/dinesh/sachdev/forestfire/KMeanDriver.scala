package indore.dinesh.sachdev.forestfire

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans

object KMeanDriver {

  def main(args: Array[String]): Unit = {
    val schema = StructType(Array(
      StructField("area", DoubleType, true),
      StructField("perimeter", DoubleType, true),
      StructField("firenum", DoubleType, true),
      StructField("fire_id", DoubleType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("date", TimestampType, true),
      StructField("julian", IntegerType, true),
      StructField("gmt", IntegerType, true),
      StructField("temp", DoubleType, true),
      StructField("spix", DoubleType, true),
      StructField("tpix", DoubleType, true),
      StructField("src", StringType, true),
      StructField("sat_src", StringType, true),
      StructField("conf", IntegerType, true),
      StructField("frp", DoubleType, true)))

    val spark: SparkSession = SparkSession.builder().appName("ForestFireModel").getOrCreate()

    import spark.implicits._
    //----------------
    val df_all = spark.read.format("csv").option("header", "true").schema(schema).load("D:\\data\\forestfire")

    // Include only fires with coordinates in the Pacific Northwest
    val df = df_all.filter($"lat" > 42).filter($"lat" < 50).filter($"lon" > -124).filter($"lon" < -110)

    val featureCols = Array("lat", "lon")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    val df2 = assembler.transform(df)

    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), 5043)

    val kmeans = new KMeans().setK(400).setFeaturesCol("features").setMaxIter(5)

    val model = kmeans.fit(trainingData)

    println("Final Centers: ")

    model.clusterCenters.foreach(println)

    // Save the model to disk
    //------------------
    model.write.overwrite().save("D:\\data\\forestfireModel")

    val test_coordinate = Seq((42.3, -112.2)).toDF("lat", "lon")
    val df3 = assembler.transform(test_coordinate)
    val categories = model.transform(df3)
    val centroid_id = categories.select("prediction").rdd.map(r =>
      r(0)).collect()(0).asInstanceOf[Int]
    
    println(centroid_id+"-------->"+model.clusterCenters(centroid_id))
  }
}