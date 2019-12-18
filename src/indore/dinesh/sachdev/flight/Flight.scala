package indore.dinesh.sachdev.flight

/* Class to define a data set*/
case class Flight(_id: String, dofW: Integer, carrier: String,
                  origin: String, dest: String, crsdephour: Integer, crsdeptime: Double, depdelay: Double, crsarrtime: Double, arrdelay: Double,
                  crselapsedtime: Double, dist: Double) extends Serializable