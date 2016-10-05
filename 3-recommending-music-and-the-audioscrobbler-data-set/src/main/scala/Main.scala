import org.apache.spark._

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Recommending Music Using ALS Collaborative Filtering")
      .setMaster("local")

    val sc = new SparkContext(conf)

    println("Hello, world!")

    sc.stop()
  }
}


