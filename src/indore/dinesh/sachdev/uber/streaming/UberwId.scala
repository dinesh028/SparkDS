package indore.dinesh.sachdev.uber.streaming

// Uber with unique Id and Cluster id and cluster lat lon
case class UberwId(_id: String, dt: java.sql.Timestamp, lat: Double, lon: Double,
                   base: String, cid: Integer, clat: Double, clon: Double) extends Serializable