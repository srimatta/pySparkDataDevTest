# H1 pySpark Data Dev Test - Taxi journeys January 2021
---

Two raw data files can be found in the rawdata folder containing trip information for NYC taxi journeys in January 2021

The aim of this excercise is to develop a simple pipeline to complete the following steps:-

1. Import the Yellow and Green data files, convert them to Parquet format and save them to the PipelineData/Bronze directory
2. Transform the datafiles to a consistant schema and save them in Parquet format to the PipelineData/Silver directory in the format specified in the table below
3. Validate the data and save the valid data in Parquet format and invalid data in csv format in the PipelineData/Silver directory 
    The validation rules mandate that all journeys must have at least 1 passenger. Journies with no passengers are not valid. All records should have a vendor id. Journies with no Vendor Id should have their ID set to 999 before saving to the valid data set.
4. Dedupe the data and save in Parquet format in the PipelineData/silver folder (Dedupe based on Pickup location, Pick up Time, Drop off time, drop off location and vendor)

The format to transform the data to is as follows:-

|Silver Table     |           Green                 |                         Yellow|
|-----------------|---------------------------------|-------------------------------|
|VendorId         |           VendorID              |          VendorID             |
|PickUpDateTime   |           lpep_pickup_datetime  |          tpep_pickup_datetime |
|DropOffDateTime  |           lpep_dropoff_datetime |          tpep_dropoff_datetime|
|PickUpLocationId |           PULocationID          |          PULocationID         |
|DropOffLocationId|           DOLocationID          |          DOLocationID         |
|PassengerCount   |           passenger_count       |          passenger_count      |
|TripDistance     |           trip_distance         |          trip_distance        |
|TipAmount        |           tip_amount            |          tip_amount           |
|TotalAmount      |           total_amount          |          total_amount         |

5. Shape the data to output csv files to the Gold folder containing the following:-
    File 1  - Locations
    The total fares by pickup location
    The total tips by pickup location
    The average distance by pickup location
    The average distance by dropoff location
    File 2 - Vendors
    The total fare by vendor
    The total tips by vendor
    The average fare by vendor
    The average tips by vendor
6. Import these files to PostgresSQL in new tables using the docker details provided below

## H2 Setting up your enviroment

[jupyter/pyspark docker] [https://hub.docker.com/r/jupyter/pyspark-notebook]

Please install docker on your machine and run docker-compose up from this directory to create a container to complete the excercise.
The output log will provide a url to access your jupyter/pyspark-notebook:latest development environment. 

[postgres docker] [https://hub.docker.com/_/postgres]
To create the docker container required for the excercise please open a command termial and navigate to the postgres folder. From this folder run docker-compose up
This will allow you to connect to the postgres DB using the following credentials:-
username:- postgres
password:- example
host:- 127.0.0.1
port:- 5432

If you do not have a client to access the Postgres already please either download PGAdmin or use Adminer which is included in the container running in the following location:- locahost:8080
To access the DB in Adminer please use the following settings:-
System - Postgres
Server - db
User - postgres
Password - example
Database - postgres

## Running application

**python/taxi_journeys_etl.py** is the entry point for the application.

Note: application needs postgresql-42.3.0.jar.