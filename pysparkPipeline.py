from pyspark.sql import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def read_input_files(local_files: str):
    df = spark.read.csv(f'{local_files}', recursiveFileLookup=True, header=True)
    return df

def example_transform_drop_columns(df, columns):
    return df.drop(*columns_to_drop)

def filter_df(df):
    df = df.filter(df.start_station_id.isNotNull())
    df = df.filter(df.end_station_id.isNotNull())
    return df

def example_cast_transform(df):
    df = df.withColumn("end_station_id", df&#91;"end_station_id"].cast("int"))
    return df

columns_to_drop = &#91;'start_station_name', 'end_station_name', 'started_at', 'ended_at']
features = &#91;"rideable_type", "start_station_id", "member_casual"]

spark = SparkSession.builder.appName("my spark").master("local&#91;3]").getOrCreate()
df = read_input_files('*.csv')
df = example_transform_drop_columns(df, columns_to_drop)
df = filter_df(df)
df = example_cast_transform(df)

indexer = &#91;StringIndexer(inputCol=column, outputCol=f"{column}_index") for column in features]
pipeline = Pipeline(stages=indexer)
df = pipeline.fit(df).transform(df)

assembler = VectorAssembler(
    inputCols=&#91;"rideable_type_index", "start_station_id_index", "member_casual_index"],
    outputCol="features")

output = assembler.transform(df)

(trainingData, testData) = output.randomSplit(&#91;0.7, 0.3], seed = 100)

lr = LogisticRegression(featuresCol = 'features', labelCol = 'end_station_id', maxIter=20)

pipeline = Pipeline(stages=&#91;lr])
model = pipeline.fit(output)
predictions = model.transform(testData)

evaluator = MulticlassClassificationEvaluator(labelCol="end_station_id", predictionCol="prediction")
evaluator.evaluate(predictions)
