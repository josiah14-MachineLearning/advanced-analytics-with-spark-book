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

    /***************************************************************************
     * For each artist (id, name) pair, ensure a `name` is available and the
     * `id` is a valid Integer. Use the `Option` type to indicate `name`
     * presence.
     */
    val artistById: RDD[(Int, String)] = rawArtistData.flatMap { line =>
      val (id: String, name: String) = line.span(_ != '\t')
      name.isEmpty match {
        case true  => None
        case false => {
          val result: Try[Option[(Int, String)]] =
            Try(Some((id.toInt, name.trim))) recoverWith {
              e: Throwable => e match {
                case e: NumberFormatException => Success(None)
                case e: Throwable             => Failure(e)
            }}
          result.get
        }
      }
    }

    /***************************************************************************
     * Maps irregular artist ID to misspelled/irregular artist names to the ID
     * of the artist's "canonical" name. It has both artist ID's in each line
     * and is tab delimited.
     */
    val rawArtistAlias: RDD[StringOps] =
      sc.textFile("hdfs:///user/josiah/profiledata_06-May-2005/artist_alias.txt", 16)
        .map(augmentString)

    /***************************************************************************
     * Some lines are missing the first Artist ID.  Use the `Option` type to
     * indicate whether the first ID is present.
     *
     * Also, collect as a Map to 'map' "bad" artist IDs to their "true" ones.
     * The `collectAsMap` function skips `None` values.
     */
    val artistAlias: Map[Int, Int] = rawArtistAlias.flatMap { line =>
      val tokens: ArrayOps[StringOps] = line.split('\t').map(augmentString)

      tokens(0).isEmpty match {
        case true  => None
        case false => Some((tokens(0).toInt, tokens(1).toInt))
      }
    }.collectAsMap()

    /***************************************************************************
     * Begin building the Model.  First, convert all artist IDs to their
     * canonical ID. Because of the MLlib API, artists will be referred to as
     * "products".  users remain "users".
     **************************************************************************/

    /***************************************************************************
     * "Broadcast" the "bad" -> "good" ID mapping to make it available to all
     * of the Spark tasks as a shared immutable on each executor (each executor
     * gets its own copy).
     *
     * (Thousands of tasks per executor may run, but the number of executors
     * tends to be small).
     */
    val bArtistAlias: Broadcast[Map[Int, Int]] = sc.broadcast(artistAlias)

    /***************************************************************************
     * For all records in the `userArtistData` scores data, convert the "bad"
     * artist IDs to "good" ones and return the results as a distributed
     * collection of mllib.recommendation.Ratings.
     *
     * The result should be cached because the ALS algorithm is iterative and
     * will need access to this training data many times.  `cache`ing reduces
     * the chances that Spark will have to recompute the data set each time it
     * is accessed by the ALS algorithm.
     *
     * Use the Storage tab in the Spark UI to see how much of the RDD is cached
     * and its impact on the RAM usage.
     */
    val trainData: RDD[Rating] = rawUserArtistData.map { line =>
      val Array(userId: Int, artistId: Int, count: Int) =
        line.split(' ').map(_.toInt)
      val finalArtistId: Int = bArtistAlias.value.getOrElse(artistId, artistId)
      Rating(userId, finalArtistId, count.toDouble)
    }.cache()

    // 10 features for each user and artist, the rest of the numbers are unexplained,
    // but are parameters for tuning the execution of the ALS algorithm.
    val model: MatrixFactorizationModel = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

    println("ARTIST" + artistById.lookup(6803336).head)
    println("ARTIST" + artistById.lookup(1000010).head)
    val newModel = model.userFeatures.mapValues(_.mkString(", ")).cache()
    println("FEATURES" + newModel.first())
    newModel.top(10).foreach(println)

    // For user 2093760, find all of the artists s/he rated.

    val rawArtistsForUser: RDD[Rating] = trainData.filter {
      case Rating(user, _, _) => user == 2093760
    }

    val existingProducts: Set[Int] = rawArtistsForUser.map {
      case Rating(_, artist, _) => artist
    }.collect().toSet

    artistById.filter { case (id, name) =>
      existingProducts.contains(id)
    }.values.collect().foreach(println)

    // And then recommend 5 new artists for user 2093760.
    val recommendations = model.recommendProducts(2093760, 5)
    recommendations.foreach(println)

    sc.stop()
  }
}
