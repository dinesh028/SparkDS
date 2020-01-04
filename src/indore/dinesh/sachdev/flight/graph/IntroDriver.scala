package indore.dinesh.sachdev.flight.graph
import org.apache.log4j.{ Level, Logger }

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.types.StructType
import org.graphframes._
import org.apache.log4j.Logger
import org.apache.log4j.Level

object IntroDriver {
  def main(args: Array[String]): Unit = {
    // val rootLogger = Logger.getRootLogger()
    //rootLogger.setLevel(Level.ERROR)

    // Logger.getLogger("org.apache.spark").setLevel(Level.DEBUG)
    //Logger.getLogger("org.spark-project").setLevel(Level.DEBUG)

    val spark = SparkSession.builder().enableHiveSupport().getOrCreate()
    import spark.implicits._

    val airports = Array(Airport("SFO", "San Francisco"), Airport("ORD", "Chicago"), Airport("DFW", "Dallas Fort Worth"))
    val vertices = spark.createDataset(airports).toDF
    vertices.show

    val flights = Array(Flight("SFO_ORD_2017-01-01_AA", "SFO", "ORD", 1800, 40), Flight("ORD_DFW_2017-01-01_UA", "ORD", "DFW", 800, 0), Flight("DFW_SFO_2017-01-01_DL", "DFW", "SFO", 1400, 10))

    val edges = spark.createDataset(flights).toDF
    edges.show

    // define the graph
    val graph = GraphFrame(vertices, edges)

    // show graph vertices
    graph.vertices.show

    // show graph edges
    graph.edges.show

    // How many airports?
    graph.vertices.count

    // How many flights?
    graph.edges.count

    //Which flight routes are greater than 1000 miles in distance?
    graph.edges.filter($"dist" > 1000).show

    //The GraphFrames triplets put all of the edge, src, and dst columns together in a DataFrame.
    graph.triplets.printSchema
    graph.triplets.show

    // print out longest routes
    graph.edges.groupBy("src", "dst").max("dist").sort(desc("max(dist)")).show
  }
}