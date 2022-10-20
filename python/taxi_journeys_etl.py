from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

import pyspark.sql.functions as F


def rename_columns(df, columns):
    if isinstance(columns, dict):
        return df.select(*[F.col(col_name).alias(columns.get(col_name, col_name)) for col_name in df.columns])
    else:
        raise ValueError("columns should be a dict, like {'old_name_1':'new_name_1', 'old_name_2':'new_name_2'}")


def write_to_postgres(dataframe, tablename, url, user, password):
    dataframe.write.mode('append').format("jdbc"). \
        option("driver", "org.postgresql.Driver"). \
        option("url", url). \
        option("dbtable", tablename). \
        option("user", user). \
        option("password", password). \
        save()


def create_ingestion_data(spark, schema, args):
    '''
    create_ingestion_data creates bronze data
    :param spark: spark session
    :param schema:  raw schema
    :param args:
    :return:
    '''
    green_trips_data_df = spark.read.option('header', True).schema(schema).csv(args.green_trips_data_path)
    yellow_trips_data_df = spark.read.option('header', True).schema(schema).csv(args.yellow_trips_data_path)

    green_trips_data_df.write.mode('append').parquet(args.bronze_path)
    yellow_trips_data_df.write.mode('append').parquet(args.bronze_path)


def create_refined_data(spark, schema, args, number_of_files=1):
    '''
    create_refined_data creates silver data
    :param spark: spark session
    :param schema:  raw schema
    :param args:
    :param number_of_files: number of output files to write
    :return:
    '''
    columns_map = {'VendorID': 'VendorId', 'lpep_pickup_datetime': 'PickUpDateTime',
                   'lpep_dropoff_datetime': 'DropOffDateTime',
                   'PULocationID': 'PickUpLocationId', 'DOLocationID': 'DropOffLocationId',
                   'passenger_count': 'PassengerCount',
                   'trip_distance': 'TripDistance', 'tip_amount': 'TipAmount', 'total_amount': 'TotalAmount'}

    raw_data_df = spark.read.schema(schema).parquet(args.bronze_path)
    df = rename_columns(raw_data_df, columns_map)

    invalid_data = df.filter(F.col('passenger_count') < 1)
    invalid_data.write.mode('append').option('header', True).csv(args.silver_path + '/invalid')

    valid_data = df.filter(F.col('passenger_count') > 0).na.fill(value=999, subset=['VendorId']). \
        drop_duplicates(['PickUpLocationId', 'PickUpDateTime', 'DropOffDateTime', 'DropOffLocationId', 'VendorId'])
    valid_data.coalesce(number_of_files).write.mode('append').parquet(args.silver_path + '/valid')


def create_aggregate_data(spark, schema, args, number_of_files=1):
    '''
    create_aggregate_data creates gold data
    :param spark:  spark session
    :param schema: silver schema
    :param args:
    :param number_of_files: number of output files to write
    :return:
    '''
    valid_data = spark.read.schema(schema).parquet(args.silver_path + '/valid')

    pickUpWindowSpec = Window.partitionBy('PickUpLocationId')
    dropOffWindowSpec = Window.partitionBy('DropOffLocationId')

    locations = valid_data. \
        withColumn('total_fares_by_pickup_location', F.sum(F.col('TotalAmount')).over(pickUpWindowSpec)). \
        withColumn('total_tips_by_pickup_location', F.sum(F.col('TipAmount')).over(pickUpWindowSpec)). \
        withColumn('avg_distance_by_pickup_location', F.avg(F.col('TripDistance')).over(pickUpWindowSpec)). \
        withColumn('avg_distance_by_dropoff_location', F.avg(F.col('TripDistance')).over(dropOffWindowSpec)). \
        select('total_fares_by_pickup_location', 'total_tips_by_pickup_location', 'avg_distance_by_pickup_location',
               'avg_distance_by_dropoff_location')

    vendorsWindowSpec = Window.partitionBy('VendorId')

    vendors = valid_data. \
        withColumn('total_fares_by_vendor', F.sum(F.col('TotalAmount')).over(vendorsWindowSpec)). \
        withColumn('total_tips_by_vendor', F.sum(F.col('TipAmount')).over(vendorsWindowSpec)). \
        withColumn('avg_fares_by_vendor', F.avg(F.col('TotalAmount')).over(vendorsWindowSpec)). \
        withColumn('avg_tips_by_vendor', F.avg(F.col('TipAmount')).over(vendorsWindowSpec)). \
        select('total_fares_by_vendor', 'total_tips_by_vendor', 'avg_fares_by_vendor', 'avg_tips_by_vendor')

    locations.coalesce(number_of_files).write.mode('append').parquet(args.gold_path + '/locations')
    vendors.coalesce(number_of_files).write.mode('append').parquet(args.gold_path + '/vendors')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        default='local',
                        help="spark running mode")

    parser.add_argument("--application_name",
                        default='TaxiJourneysETL',
                        help="application name")

    parser.add_argument("--postgres_jars_path",
                        default='../lib/postgresql-42.3.0.jar',
                        help="postgres jars path")

    parser.add_argument("--green_trips_data_path",
                        default='../RawData/green_tripdata_2021-01.csv',
                        help="green trips data path")

    parser.add_argument("--yellow_trips_data_path",
                        default='../RawData/yellow_tripdata_2021-01.csv',
                        help="yellow trips data path")

    parser.add_argument("--bronze_path",
                        default='../PipelineData/Bronze',
                        help="bronze directory location")

    parser.add_argument("--silver_path",
                        default='../PipelineData/Silver',
                        help="silver directory location")

    parser.add_argument("--gold_path",
                        default='../PipelineData/Gold',
                        help="gold directory path")

    parser.add_argument("--jdbc_url",
                        default='jdbc:postgresql://localhost:5432/',
                        help="jdbc url")

    parser.add_argument("--user_name",
                        default='postgres',
                        help="user name")

    parser.add_argument("--password",
                        default='example',
                        help="password")

    args = parser.parse_args()
    print("args : ", args)

    # Creating SparkSession
    spark = SparkSession.builder \
        .master(args.mode) \
        .appName(args.application_name) \
        .config('spark.jars', args.postgres_jars_path) \
        .getOrCreate()

    raw_schema = StructType([ \
        StructField('VendorID', IntegerType(), True), \
        StructField('lpep_pickup_datetime', StringType(), True), \
        StructField('lpep_dropoff_datetime', StringType(), True), \
        StructField('PULocationID', IntegerType(), True), \
        StructField('DOLocationID', IntegerType(), True), \
        StructField('passenger_count', IntegerType(), True), \
        StructField('trip_distance', DoubleType(), True),
        StructField('tip_amount', DoubleType(), True), \
        StructField('total_amount', DoubleType(), True) \
        ])

    silver_schema = StructType([ \
        StructField('VendorId', IntegerType(), True), \
        StructField('PickUpDateTime', StringType(), True), \
        StructField('DropOffDateTime', StringType(), True), \
        StructField('PickUpLocationId', IntegerType(), True), \
        StructField('DropOffLocationId', IntegerType(), True), \
        StructField('PassengerCount', IntegerType(), True), \
        StructField('TripDistance', DoubleType(), True),
        StructField('TipAmount', DoubleType(), True), \
        StructField('TotalAmount', DoubleType(), True) \
        ])

    # creating bronze data
    create_ingestion_data(spark, raw_schema, args)
    # creating silver data
    create_refined_data(spark, raw_schema, args)
    # creating gold data
    create_aggregate_data(spark, silver_schema, args)

    locations = spark.read.parquet(args.gold_path + '/locations')
    vendors = spark.read.parquet(args.gold_path + '/vendors')

    write_to_postgres(locations,
                      'locations',
                      args.jdbc_url,
                      args.user_name,
                      args.password)
    write_to_postgres(vendors,
                      'vendors',
                      args.jdbc_url,
                      args.user_name,
                      args.password)
