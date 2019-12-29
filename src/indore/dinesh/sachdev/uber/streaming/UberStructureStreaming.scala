package indore.dinesh.sachdev.uber.streaming

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.clustering._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.client.Put
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.mapred.TableOutputFormat
import org.apache.hadoop.mapred.JobConf
import org.apache.hadoop.hbase.client.HTable
import org.apache.hadoop.hbase.client.ConnectionFactory
import org.apache.hadoop.hbase.TableName
import org.apache.hadoop.hbase.client.Table

object UberStructureStreaming {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("uber").getOrCreate()

    import spark.implicits._
    // load the saved model from the distributed file system
    val model = KMeansModel.load("/data/uber")

    var ac = new Array[Center](20)

    var index: Int = 0

    model.clusterCenters.foreach(x => {
      ac(index) = Center(index, x(0), x(1));
      index += 1;
    })

    val ccdf = spark.createDataset(ac)
    ccdf.show()

    val df1 = spark.readStream.format("kafka")
      .option("kafka.bootstrap.servers", "sandbox-hdp.hortonworks.com:6667")
      .option("subscribe", "kafkatopic")
      .option("group.id", "testgroup")
      .option("startingOffsets", "latest")
      .option("failOnDataLoss", false)
      .option("maxOffsetsPerTrigger", 1000)
      .load()

    df1.printSchema()

    // Parse string into Uber case class
    def parseUber(str: String): Uber = {
      val p = str.split(",")
      Uber(p(0), p(1).toDouble, p(2).toDouble, p(3), p(4))
    }

    spark.udf.register("deserialize", (message: String) => parseUber(message))

    val df2 = df1.selectExpr("deserialize(CAST(value as STRING)) AS message").select($"message".as[Uber])

    //Vector assembler
    val featureCols = Array("lat", "lon")
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val df3 = assembler.transform(df2)

    //use model to get the clusters from the features
    val clusters1 = model.transform(df3)

    val temp = clusters1.select($"dt".cast(TimestampType), $"lat", $"lon", $"base", $"rdt", $"cid")

    val clusters = temp.join(ccdf, Seq("cid")).as[UberC]

    // enrich with unique id for Mapr-DB
    def createUberwId(uber: UberC): UberwId = {
      val id = uber.cid + "_" + uber.rdt
      UberwId(id, uber.dt, uber.lat, uber.lon, uber.base, uber.cid, uber.clat, uber.clon)
    }

    val cdf: Dataset[UberwId] = clusters.map(uber =>
      createUberwId(uber))

    //This example print it to console. But, we can actually put it to other data sources.
    //val streamingquery = cdf.writeStream.outputMode("Append").format("console").start.awaitTermination()

    val query = cdf.writeStream.foreach(new myWriter())

    query.start.awaitTermination()

  }

}

class myWriter extends ForeachWriter[UberwId] with Serializable {

  var table: Table = null
  def open(partitionId: Long, version: Long): Boolean = {
    @transient val conf = HBaseConfiguration.create()
    conf.set("zookeeper.znode.parent", "/hbase-unsecure")
    conf.set("hbase.zookeeper.property.clientPort", "2181")
    conf.set("hbase.zookeeper.quorum", "sandbox-hdp.hortonworks.com")  
    @transient val connection = ConnectionFactory.createConnection(conf)
    
    val table1 = TableName.valueOf("uber")
    table = connection.getTable(table1)
    true
  }

  def process(record: UberwId): Unit = {
    table.put(convertToPut(record)._2)
  }

  def close(errorOrNull: Throwable): Unit = {
    table.close()
  }

  def convertToPut(uber: UberwId): (ImmutableBytesWritable, Put) = {
    val cfDataBytes = Bytes.toBytes("cf")

    val put = new Put(Bytes.toBytes(uber._id))
    put.addImmutable(cfDataBytes, Bytes.toBytes("dt"), Bytes.toBytes(uber.dt.toString()));
    // add to column family data, column  data values to put object
    return (new ImmutableBytesWritable(Bytes.toBytes(uber._id)), put)
  }
}