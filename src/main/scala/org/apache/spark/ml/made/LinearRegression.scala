package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{
  HasFeaturesCol,
  HasLabelCol,
  HasMaxIter,
  HasPredictionCol,
  HasStepSize
}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

import breeze.linalg.{sum, DenseVector => BreezeDenseVector}

trait LinearRegressionParams
    extends HasLabelCol
    with HasFeaturesCol
    with HasPredictionCol
    with HasMaxIter
    with HasStepSize {

  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  setDefault(maxIter -> 1000, stepSize -> 0.1)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(
        schema,
        schema(getFeaturesCol).copy(name = getPredictionCol)
      )
    }
  }
}

class LinearRegression(override val uid: String)
    extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setStepSize(value: Double): this.type = set(stepSize, value)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val datasetExt: Dataset[_] = dataset.withColumn("ones", lit(1))
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array($(featuresCol), "ones", $(labelCol)))
      .setOutputCol("features_ext")

    val vectors: Dataset[Vector] = assembler
      .transform(datasetExt)
      .select("features_ext")
      .as[Vector]

    val numFeatures: Int = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var w: BreezeDenseVector[Double] =
      BreezeDenseVector.rand[Double](numFeatures + 1)

    for (_ <- 0 until $(maxIter)) {
      val summary = vectors.rdd
        .mapPartitions((data: Iterator[Vector]) => {
          val summarizer = new MultivariateOnlineSummarizer()
          data.foreach(vector => {
            val X = vector.asBreeze(0 until w.size).toDenseVector
            val y = vector.asBreeze(w.size)
            summarizer.add(fromBreeze(X * (sum(X * w) - y)))
          })
          Iterator(summarizer)
        })
        .reduce(_ merge _)
      w = w - $(stepSize) * summary.mean.asBreeze
    }

    copyValues(
      new LinearRegressionModel(
        Vectors.fromBreeze(w(0 until w.size - 1)).toDense,
        w(w.size - 1)
      )
    ).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] =
    defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made] (
    override val uid: String,
    val w: DenseVector,
    val b: Double
) extends Model[LinearRegressionModel]
    with LinearRegressionParams
    with MLWritable {

  private[made] def this(w: DenseVector, b: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), w.toDense, b)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(w, b),
    extra
  )

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(
      uid + "_transform",
      (x: Vector) => {
        Vectors.fromBreeze(
          BreezeDenseVector(w.asBreeze.dot(x.asBreeze) + b)
        )
      }
    )

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Vector) =
        w.asInstanceOf[Vector] -> Vectors.fromBreeze(
          BreezeDenseVector(b)
        )

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] =
    new MLReader[LinearRegressionModel] {
      override def load(path: String): LinearRegressionModel = {
        val metadata = DefaultParamsReader.loadMetadata(path, sc)

        val vectors = sqlContext.read.parquet(path + "/vectors")

        implicit val encoder: Encoder[Vector] = ExpressionEncoder()

        val (w, b) = vectors
          .select(vectors("_1").as[Vector], vectors("_2").as[Vector])
          .first()

        val model = new LinearRegressionModel(w.toDense, b(0))
        metadata.getAndSetParams(model)
        model
      }
    }
}
