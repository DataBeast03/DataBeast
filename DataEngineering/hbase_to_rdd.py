import happybase
import pyspark
from pyspark.sql.types import *


def get_userName_data(table):
    return [(data[0], data[1]["d:name"], data[1]["d:ts"]) for data in table.scan()]

def get_userGender_data(table):
    return [(data[0], data[1]["d:gender"], data[1]["d:ts"]) for data in table.scan()]

def get_userLocation_data(table):
    return [(data[0], data[1]["d:city"],data[1]["d:state"], data[1]["d:country"],data[1]["d:ts"]) for data in table.scan()]

def get_purchaseEdge_data(table):
    # userId,itemId,ts
    return [(data[0], data[1]["d:item_id"], data[1]["d:ts"]) for data in table.scan()]

def get_ratingEdge_data(table):
    # userId, itemdid, rating, ts, review
    return [(data[0], data[1]["d:item_id"], data[1]["d:rating"], data[1]["d:ts"], data[1]["d:review"]) for data in table.scan()]

def get_itemName_data(table):
    # userId, itemdid, rating, ts, review
    return [(data[0], data[1]["d:name"], data[1]["d:ts"],) for data in table.scan()]


def get_current_property(values):
    '''scans property time stampts and selects the most current property '''
    timestamp = -1
    timestamps = dict()
    
    for prop in values:
        timestamps[prop[timestamp]]=prop[:timestamp]
    max_ts = max(timestamps.keys())
    
    if len(timestamps[max_ts]) == 1:
        return (timestamps[max_ts][0],max_ts)
    elif len(timestamps[max_ts]) == 2:
        return (timestamps[max_ts][0],timestamps[max_ts][1],max_ts)
    else:
        return (timestamps[max_ts][0],timestamps[max_ts][1],timestamps[max_ts][2],max_ts)


def create_table_schemas():
    # create sparksql schemas for tables
    schema_gender = StructType( [
    StructField('user_id',StringType(),True),
    StructField('gender',StringType(),True),
    StructField('timestamp',StringType(),True),] )

    schema_location = StructType( [
    StructField('user_id',StringType(),True),
    StructField('city',StringType(),True),
    StructField('state',StringType(),True),
    StructField('country',StringType(),True),
    StructField('timestamp',StringType(),True),] )

    schema_name = StructType( [
    StructField('user_id',StringType(),True),
    StructField('name',StringType(),True),
    StructField('timestamp',StringType(),True),] )

    schema_review = StructType( [
    StructField('user_id',StringType(),True),
    StructField('item_id',StringType(),True),
    StructField('rating',StringType(),True),
    StructField('timestamp',StringType(),True),
    StructField('review',StringType(),True)] )

    schema_item_name = StructType( [
    StructField('item_id',StringType(),True),
    StructField('name',StringType(),True),
    StructField('timestamp',StringType(),True)] )

    schema_purchase = StructType( [
    StructField('user_id',StringType(),True),
    StructField('item_id',StringType(),True),
    StructField('timestamp',StringType(),True)] )


def get_names():  
    table = connection.table("user_name")
    user_name = sc.parallelize(get_userName_data(table))

    normalized_names = user_name.map(lambda (user_id, name, ts): (user_id, (name, ts)))\
                            .groupByKey()\
                            .mapValues(lambda line: get_current_property(line))\
                            .map(lambda line: (line[0], line[1][0], line[1][1]))
    return normalized_names


def get_genders():
    table = connection.table("gender")
    gender = sc.parallelize(get_userGender_data(table))

    normalized_genders = gender.map(lambda (user_id, name, ts): (user_id, (name, ts)))\
                            .groupByKey()\
                            .mapValues(lambda line: get_current_property(line))\
                            .map(lambda line: (line[0], line[1][0], line[1][1]))
    return normalized_genders

def get_locations():
    table = connection.table("location")
    gender = sc.parallelize(get_userGender_data(table))

    normalized_location = location.map(lambda (user_id, name, ts): (user_id, (name, ts)))\
                            .groupByKey()\
                            .mapValues(lambda line: get_current_property(line))\
                            .map(lambda line: (line[0], line[1][0], line[1][1]))
    return normalized_location


def get_review_edges():
    table = connection.table("review_edge")
    review_edge = sc.parallelize(get_ratingEdge_data(table))

def get_purchase_edges():
    table = connection.table("purchase_edge")
    purchase_edge = sc.parallelize(get_purchaseEdge_data(table))

def get_item_names():
    table = connection.table("item_name")
    item_name = sc.parallelize(get_itemName_data(table))

if __name__ == "__main__":
    '''creates spark, sparksql contexts, & hbase connection'''
    connection = happybase.Connection(host = 'localhost')
    print "hbase connection: {}".format(connection) 
    sc = pyspark.SparkContext()
    print "Spark Context: {}".format(sc)
    sqlContext = pyspark.HiveContext(sc)
    print "SparkSQL Context: {}".format(sqlContext)

    create_table_schemas()
    print "table schemas created"

    while True:
        yield get_names()
        yield get_genders()
        yield get_locations()
        yield get_review_edges()
        yield get_purchase_edges()
        yield get_item_names

    


