package indore.dinesh.sachdev.uber

case class Uber(dt: java.sql.Timestamp, lat: Double,
                lon: Double, base: String) extends Serializable