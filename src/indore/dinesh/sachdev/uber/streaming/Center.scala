package indore.dinesh.sachdev.uber.streaming

// create a Dataset with cluster id and location
case class Center(cid: Integer, clat: Double,
                  clon: Double)