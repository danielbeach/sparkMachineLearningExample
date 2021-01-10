from pyspark.sql import *
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import keyword_only


class CustomTransform(Transformer):
    @keyword_only
    def __init__(self):
        """Initialize."""
        super(CustomTransform, self).__init__()

    def _transform(self, dataframe):
        df = dataframe.drop(*['start_station_name', 'end_station_name', 'started_at', 'ended_at'])
        df = df.filter(df.start_station_id.isNotNull())
        df = df.filter(df.end_station_id.isNotNull())
        df = df.withColumn("end_station_id", df["end_station_id"].cast("int"))
        return df


def feature_transforms(data_frame: DataFrame, get_columns_to_drop: list) -> DataFrame:
    df = data_frame.drop(*get_columns_to_drop)
    df = df.filter(df.start_station_id.isNotNull())
    df = df.filter(df.end_station_id.isNotNull())
    df = df.withColumn("end_station_id", df["end_station_id"].cast("int"))
    return df


def read_input_files(local_files: str):
    df = spark.read.csv(f'{local_files}', recursiveFileLookup=True, header=True)
    return df

columns_to_drop = ['start_station_name', 'end_station_name', 'started_at', 'ended_at']
features = ["rideable_type", "start_station_id", "member_casual"]

spark = SparkSession.builder.appName("my ml pipeline").master("local[3]").getOrCreate()
df = read_input_files('*.csv')

steps = [CustomTransform()]
indexer_steps = [StringIndexer(inputCol=column, outputCol=f"{column}_index") for column in features]
steps.extend(indexer_steps)

pipeline = Pipeline(stages=steps)

df = pipeline.fit(df).transform(df)

assembler = VectorAssembler(
    inputCols=["rideable_type_index", "start_station_id_index", "member_casual_index"],
    outputCol="features")

output = assembler.transform(df)

(trainingData, testData) = output.randomSplit([0.7, 0.3], seed = 100)

lr = LogisticRegression(featuresCol = 'features', labelCol = 'end_station_id', maxIter=20)

pipeline = Pipeline(stages=[lr])
model = pipeline.fit(output)
predictions = model.transform(testData)

evaluator = MulticlassClassificationEvaluator(labelCol="end_station_id", predictionCol="prediction")
evaluator.evaluate(predictions)
print(predictions.head(100))
