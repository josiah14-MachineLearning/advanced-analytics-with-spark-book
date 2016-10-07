import org.apache.spark._
import scala.util.{Try, Success, Failure}
import org.apache.spark.mllib.recommendation._

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Recommending Music Using ALS Collaborative Filtering")
      .setMaster("local")

    val sc = new SparkContext(conf)

    val rawUserArtistData = sc.textFile("hdfs:///user/josiah/profiledata_06-May-2005/user_artist_data.txt", 16)
    val rawArtistData = sc.textFile("hdfs:///user/josiah/profiledata_06-May-2005/artist_data.txt", 16)
    val artistById = rawArtistData.flatMap { line =>
      val (id, name) = line.span(_ != '\t')
      name.isEmpty match {
        case true  => None
        case false => {
          val result = Try(Some((id.toInt, name.trim))) recoverWith { e: Throwable => e match {
            case e: NumberFormatException => Success(None)
            case e: Throwable             => Failure(e)
          }}
          result.get
        }
      }
    }
    val rawArtistAlias = sc.textFile("hdfs:///user/josiah/profiledata_06-May-2005/artist_alias.txt", 16)
    val artistAlias = rawArtistAlias.flatMap { line =>
      val tokens = line.split('\t')
      tokens(0).isEmpty match {
        case true  => None
        case false => Some((tokens(0).toInt, tokens(1).toInt))
      }
    }.collectAsMap()

    val bArtistAlias = sc.broadcast(artistAlias)

    val trainData = rawUserArtistData.map { line =>
      val Array(userId, artistId, count) = line.split(' ').map(_.toInt)
      val finalArtistId = bArtistAlias.value.getOrElse(artistId, artistId)
      Rating(userId, finalArtistId, count)
    }.cache()

    val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

    println(artistById.lookup(6803336).head)
    println(artistById.lookup(1000010).head)
    println(model.userFeatures.mapValues(_.mkString(", ")).first())

    sc.stop()
  }
}
