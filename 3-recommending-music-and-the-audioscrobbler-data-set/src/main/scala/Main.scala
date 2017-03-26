import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import scala.util.{Try, Success, Failure}
import org.apache.spark.mllib.recommendation._
import scala.collection.Map
import scala.collection.immutable.StringOps
import scala.collection.mutable.ArrayOps
import scala.Option.option2Iterable
import scala.{Array, Option}

object Main {
  def main(args: Array[String]) {
    val conf: SparkConf = new SparkConf()
        .setAppName("Recommending Music Using ALS Collaborative Filtering")
        .setMaster("local")

    val sc: SparkContext = new SparkContext(conf)

    val rawUserArtistData: RDD[String] =
      sc.textFile("hdfs:///user/josiah/profiledata_06-May-2005/user_artist_data.txt", 16)
    val rawArtistData: RDD[String] =
      sc.textFile("hdfs:///user/josiah/profiledata_06-May-2005/artist_data.txt", 16)

    val artistById: RDD[(Int, String)] = rawArtistData.flatMap { line =>
      val (id: String, name: String) = line.span(_ != '\t')
      name.isEmpty match {
        case true  => option2Iterable(None)
        case false => {
          val result: Try[Option[(Int, String)]] =
            Try(Some((augmentString(id).toInt, name.trim))) recoverWith {
              e: Throwable => e match {
                case e: NumberFormatException => Success(None)
                case e: Throwable             => Failure(e)
            }}
          option2Iterable(result.get)
        }
      }
    }

    val rawArtistAlias: RDD[StringOps] =
      sc.textFile("hdfs:///user/josiah/profiledata_06-May-2005/artist_alias.txt", 16)
        .map(augmentString(_))

    val artistAlias: Map[Int, Int] = rawArtistAlias.flatMap { line =>
      val tokens: ArrayOps[StringOps] =
        refArrayOps(line.split('\t')).map(s => augmentString(s))

      tokens(0).isEmpty match {
        case true  => option2Iterable(None)
        case false => option2Iterable(Some((tokens(0).toInt, tokens(1).toInt)))
      }
    }.collectAsMap()

    val bArtistAlias: Broadcast[Map[Int, Int]] = sc.broadcast(artistAlias)

    val trainData: RDD[Rating] = rawUserArtistData.map { line =>
      val Array(userId: Int, artistId: Int, count: Int) =
        line.split(' ').map(_.toInt)
      val finalArtistId: Int = bArtistAlias.value.getOrElse(artistId, artistId)
      Rating(userId, finalArtistId, count.toDouble)
    }.cache()

    val model: MatrixFactorizationModel = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

    println("ARTIST" + artistById.lookup(6803336).head)
    println("ARTIST" + artistById.lookup(1000010).head)
    println("FEATURES" + model.userFeatures.mapValues(_.mkString(", ")).first())

    sc.stop()
  }
}
