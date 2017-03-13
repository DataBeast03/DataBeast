# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class WebscraperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    title = scrapy.Field()
    url = scrapy.Field()
    company = scrapy.Field()
    sql = scrapy.Field()
    python = scrapy.Field()
    ml = scrapy.Field()
    spark = scrapy.Field()
    hadoop = scrapy.Field()
    dl = scrapy.Field()
    stats = scrapy.Field()
    ts = scrapy.Field()
    de = scrapy.Field()
    dp = scrapy.Field()
    nlp = scrapy.Field()
    bd = scrapy.Field()
    pca = scrapy.Field()
    pred_modeling = scrapy.Field()
    anomaly_detection = scrapy.Field()
    data_analysis = scrapy.Field()
    exp_design = scrapy.Field()
    cnn = scrapy.Field()
    bayes_opt = scrapy.Field()
    pipeline = scrapy.Field()
