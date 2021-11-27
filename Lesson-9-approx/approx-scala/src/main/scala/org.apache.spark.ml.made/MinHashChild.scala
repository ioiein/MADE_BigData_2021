package org.apache.spark.ml.made

import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, LSH, LSHModel, MinHashLSH, MinHashLSHModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{IntParam, ParamValidators, Params}

import scala.util.Random

class MinHashChildModel(override val uid: String,
                        override val randCoefficients: Array[(Int, Int)],
                        bandWidth: Int)
  extends MinHashLSHModel(uid, randCoefficients) {
  override protected[ml] def hashFunction(elems: Vector): Array[Vector] = {
    require(elems.nonZeroIterator.nonEmpty, "Must have at least 1 non zero entry.")
    val hashValues = randCoefficients.map { case (a, b) =>
      elems.nonZeroIterator.map { case (i, _) =>
        ((1L + i) * a + b) % MinHashLSH.HASH_PRIME
      }.min.toDouble
    }
    hashValues.grouped(bandWidth).map(Vectors.dense).toArray
  }
}

trait ChildParams extends Params {
  final val bandWidth: IntParam = new IntParam(this, "bandWidth", "", ParamValidators.gt(0))
  final def getBandWidth: Int = $(bandWidth)

  setDefault(bandWidth -> 3)
}

class MinHashChild extends MinHashLSH with ChildParams {
  def setBandWidth(value: Int): this.type = set(bandWidth, value)
  override protected[ml] def createRawLSHModel(inputDim: Int): MinHashLSHModel = {
    require(inputDim <= MinHashLSH.HASH_PRIME,
      s"The input vector dimension $inputDim exceeds the threshold ${MinHashLSH.HASH_PRIME}.")
    val rand = new Random($(seed))
    val randCoefs: Array[(Int, Int)] = Array.fill($(numHashTables)) {
      (1 + rand.nextInt(MinHashLSH.HASH_PRIME - 1), rand.nextInt(MinHashLSH.HASH_PRIME - 1))
    }
    new MinHashChildModel(uid, randCoefs, $(bandWidth))
  }
}

//class TestModel extends LSHModel[TestModel]
//class Test extends LSH[TestModel]