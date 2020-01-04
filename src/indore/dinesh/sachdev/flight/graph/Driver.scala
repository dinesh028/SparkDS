package indore.dinesh.sachdev.flight.graph

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.types.StructType
import org.graphframes._
import org.graphframes.lib.AggregateMessages

object Driver {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().enableHiveSupport().getOrCreate()
    import spark.implicits._

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

    var file = "C:\\Work\\workspace\\SparkDS-master\\resources\\flightData\\flights.json"
    val df = spark.read.option("inferSchema", "false").schema(schema).json(file).as[indore.dinesh.sachdev.flight.Flight]

    //Edges
    val flights = df.withColumnRenamed("_id", "id")
      .withColumnRenamed("origin", "src")
      .withColumnRenamed("dest", "dst")
      .withColumnRenamed("depdelay", "delay")
    flights.show

    //Now, we define vertices.
    var airportsfile = "C:\\Work\\workspace\\SparkDS-master\\resources\\flightData\\airports.json"
    val airports = spark.read.json(airportsfile)
    airports.createOrReplaceTempView("airports")
    airports.show

    // define the graphframe
    val graph = GraphFrame(airports, flights)
    // show graph vertices
    graph.vertices.show(2)

    // show graph edges
    graph.edges.show(2)

    // How many airports?
    val numairports = graph.vertices.count

    // How many flights?
    val numflights = graph.edges.count

    // show the longest distance routes
    graph.edges
      .groupBy("src", "dst")
      .max("dist")
      .sort(desc("max(dist)")).show(4)

    //which flight have highest average delay?
    graph.edges.groupBy("crsdephour").avg("delay").sort(desc("avg(delay)")).show

    //which flights hours have highest average delay?
    graph.edges.groupBy("src", "dst").avg("delay").sort(desc("avg(delay)")).show

    //------------------
    //------------------
    //The degree of a vertex is the number of edges that touch the vertex.
    /*GraphFrames provides vertex inDegree, outDegree, and degree queries, which determine the number of incoming edges, outgoing edges, and total edges.*/
    graph.inDegrees.orderBy(desc("inDegree")).show(3)

    // which airport has the most outgoing flights?
    graph.outDegrees.orderBy(desc("outDegree")).show(3)

    // Define a reduce operation to compute the highest degree vertex
    graph.degrees.orderBy(desc("degree")).show()
    
    // use pageRank
   // val ranks =graph.pageRank.resetProbability(0.15).maxIter(1).run()
    
   // ranks.vertices.orderBy($"pagerank".desc).show()
    
    val AM = AggregateMessages
    val msgToSrc = AM.edge("delay")
    val agg = {graph.aggregateMessages.sendToSrc(msgToSrc).agg(avg(AM.msg).as("avgdelay")).orderBy(desc("avgdelay"))  
    }
    
    agg.show
    
  }
  
}