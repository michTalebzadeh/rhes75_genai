import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

from pyspark.sql import SQLContext, HiveContext
from config import config

def spark_session(appName):
  return SparkSession.builder \
    .appName(appName) \
    .enableHiveSupport() \
    .getOrCreate()


def sparkcontext():
  return SparkContext.getOrCreate()


def hivecontext():
  return HiveContext(sparkcontext())

def spark_session_local(appName):
    return SparkSession.builder \
        .master('local[1]') \
        .appName(appName) \
        .enableHiveSupport() \
        .getOrCreate()

def setSparkConfHive(spark):
    try:
        spark.conf.set("hive.exec.dynamic.partition", "true")
        spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
        spark.conf.set("spark.sql.orc.filterPushdown", "true")
        spark.conf.set("hive.msck.path.validation", "ignore")
        spark.conf.set("hive.metastore.authorization.storage.checks", "false")
        spark.conf.set("hive.metastore.client.connect.retry.delay", "5s")
        spark.conf.set("hive.metastore.client.socket.timeout", "1800s")
        spark.conf.set("hive.metastore.connect.retries", "12")
        spark.conf.set("hive.metastore.execute.setugi", "false")
        spark.conf.set("hive.metastore.failure.retries", "12")
        spark.conf.set("hive.metastore.schema.verification", "false")
        spark.conf.set("hive.metastore.schema.verification.record.version", "false")
        spark.conf.set("hive.metastore.server.max.threads", "100000")
        spark.conf.set("hive.metastore.authorization.storage.checks", "/usr/hive/warehouse")
        spark.conf.set("hive.stats.autogather", "true")
        spark.conf.set("hive.metastore.disallow.incompatible.col.type.changes", "false")
        spark.conf.set("set hive.resultset.use.unique.column.names", "false")
        spark.conf.set("hive.metastore.uris", "thrift://rhes75:9083")
        return spark
    except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)

def setSparkConfBQ(spark):
    try:
        spark.conf.set("GcpJsonKeyFile", config['GCPVariables']['jsonKeyFile'])
        spark.conf.set("BigQueryProjectId", config['GCPVariables']['projectId'])
        spark.conf.set("BigQueryDatasetLocation", config['GCPVariables']['datasetLocation'])
        spark.conf.set("google.cloud.auth.service.account.enable", "true")
        spark.conf.set("fs.gs.project.id", config['GCPVariables']['projectId'])
        spark.conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        spark.conf.set("fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        spark.conf.set("temporaryGcsBucket", config['GCPVariables']['tmp_bucket'])
        spark.conf.set("spark.sql.streaming.checkpointLocation", config['GCPVariables']['tmp_bucket'])
        return spark
    except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)

def setSparkConfRedis(spark):
     try:
        spark.conf.set("spark.redis.host", config['RedisVariables']['redisHost'])
        spark.conf.set("spark.redis.port", config['RedisVariables']['redisPort'])
        spark.conf.set("spark.redis.auth", config['RedisVariables']['Redis_password'])
        spark.conf.set("spark.redis.db", config['RedisVariables']['redisDB'])
        return spark
     except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)

def setSparkConfStreaming(spark):
    try:
        spark.conf.set("sparkDefaultParllelism", config['MDVariables']['sparkDefaultParallelism'])
        spark.conf.set("sparkSerializer", config['MDVariables']['sparkSerializer'])
        spark.conf.set("sparkNetworkTimeOut", config['MDVariables']['sparkNetworkTimeOut'])
        spark.conf.set("sparkStreamingUiRetainedBatches", config['MDVariables']['sparkStreamingUiRetainedBatches'])
        spark.conf.set("sparkWorkerUiRetainedDrivers",  config['MDVariables']['sparkWorkerUiRetainedDrivers'])
        spark.conf.set("sparkWorkerUiRetainedExecutors", config['MDVariables']['sparkWorkerUiRetainedExecutors'])
        spark.conf.set("sparkWorkerUiRetainedStages", config['MDVariables']['sparkWorkerUiRetainedStages'])
        spark.conf.set("sparkUiRetainedJobs", config['MDVariables']['sparkUiRetainedJobs'])
        spark.conf.set("spark.streaming.stopGracefullyOnShutdown", "true")
        spark.conf.set("spark.streaming.receiver.writeAheadLog.enable", "true")
        spark.conf.set("spark.streaming.driver.writeAheadLog.closeFileAfterWrite", "true")
        spark.conf.set("spark.streaming.receiver.writeAheadLog.closeFileAfterWrite", "true")
        spark.conf.set("spark.streaming.backpressure.enabled", "true")
        spark.conf.set("spark.streaming.receiver.maxRate", config['MDVariables']['sparkStreamingReceiverMaxRate'])
        spark.conf.set("spark.streaming.kafka.maxRatePerPartition", config['MDVariables']['sparkStreamingKafkaMaxRatePerPartition'])
        spark.conf.set("spark.streaming.backpressure.pid.minRate", config['MDVariables']['sparkStreamingBackpressurePidMinRate'])
        return spark
    except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)

def loadTableFromBQ(spark,dataset,tableName):
    try:
        read_df = spark.read. \
            format("bigquery"). \
            option("credentialsFile", config['GCPVariables']['jsonKeyFile']). \
            option("dataset", dataset). \
            option("table", tableName). \
            load()
        return read_df
    except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)

def writeTableToBQ(dataFrame,mode,dataset,tableName):
    try:
        dataFrame. \
            write. \
            format("bigquery"). \
            option("credentialsFile", config['GCPVariables']['jsonKeyFile']). \
            mode(mode). \
            option("dataset", dataset). \
            option("table", tableName). \
            save()
    except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)

def loadTableFromJDBC(spark, url, tableName, user, password, driver, fetchsize):
    try:
       read_df = spark.read. \
            format("jdbc"). \
            option("url", url). \
            option("dbtable", tableName). \
            option("user", user). \
            option("password", password). \
            option("driver", driver). \
            option("fetchsize", fetchsize). \
            load()
       return read_df
    except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)


def writeTableWithJDBC(dataFrame, url, tableName, user, password, driver, mode):
    try:
        dataFrame. \
            write. \
            format("jdbc"). \
            option("url", url). \
            option("dbtable", tableName). \
            option("user", user). \
            option("password", password). \
            option("driver", driver). \
            mode(mode). \
            save()
    except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)

def loadTableFromRedis(spark, tableName, keyColumn):
    try:
       read_df = spark.read. \
            format("org.apache.spark.sql.redis"). \
            option("table", tableName). \
            option("key.column", keyColumn). \
            option("infer.schema", True). \
            load()
       return read_df
    except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)

def writeTableToRedis(dataFrame, tableName, keyColumn, mode):
    try:
        dataFrame. \
            write. \
            format("org.apache.spark.sql.redis"). \
            option("table", tableName). \
            option("key.column", keyColumn). \
            mode(mode). \
            save()
    except Exception as e:
        print(f"""{e}, quitting""")
        sys.exit(1)
