package indore.dinesh.sachdev.uber.streaming

// Uber class with Cluster id,lat lon
case class UberC(dt: java.sql.Timestamp, lat: Double,
                 lon: Double, base: String, rdt: String, cid: Integer,
                 clat: Double, clon: Double) extends Serializable