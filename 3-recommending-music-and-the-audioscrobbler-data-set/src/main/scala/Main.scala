import org.apache.spark._
import scala.util.{Try, Success, Failure}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Recommending Music Using ALS Collaborative Filtering")
      .setMaster("local")

    val sc = new SparkContext(conf)

    val rawUserArtistData = sc.textFile("hdfs:///user/josiah/profiledata_06-May-2005/user_artist_data.txt", 8)
    // print(rawUserArtistData.map(_.split(' ')(0).toDouble).stats())
    val rawArtistData = sc.textFile("hdfs:///user/josiah/profiledata_06-May-2005/artist_data.txt", 8)
    val artistById = rawArtistData.flatMap { line =>
      val (id, name) = line.span(_ != '\t')
      name.isEmpty match {
        case true  => None
        case false => {
          val result = Try(Some((id.toInt, name.trim))) recoverWith { e: Throwable => e match {
            case e: NumberFormatException => Success(None)
            case e: Throwable             => Failure(throw e)
          }}
          result.get
        }
      }
    }

    println(artistById.lookup(6803336).head)

    sc.stop()
  }
}
