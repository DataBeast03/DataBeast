import happybase
from os import listdir
import shutil
import time


def insert_data_hbase(datum,label):
    '''inserts data into hbase in batchs - one insert per property/edge'''
    table = connection.table(label)
    if label == "user_name":
        table.put(datum[0], {'d:name'   : datum[1], 
                                  'd:ts' : str(datum[2])})
    elif label == "gender":
        table.put(datum[0], {'d:gender' : datum[1],
                                'd:ts'   : str(datum[2])})
    elif label == "location":
        table.put(datum[0], {'d:city'     : datum[1],
                                'd:state'  : datum[2],
                                'd:country': datum[3],
                                'd:ts'     : str(datum[4])})
    elif label == "purchase_edge":
        table.put(datum[0], {'d:item_id': datum[1],
                                'd:ts'     : str(datum[2])})
    elif label == "review_edge":
        table.put(datum[0], {'d:item_id': datum[1],
                                'd:rating' : str(datum[2]),
                                'd:ts'     : str(datum[3]),
                                'd:review' : str(datum[4])})
    else:
        table.put(datum[0], {'d:name'   : datum[1], 
                              'd:ts'     : str(datum[2])})








if __name__ == "__main__":

	connection = happybase.Connection(host = 'localhost')

	while True:
	    try:   
	        # gets folder with sub folders of each batch RDDs data
	        path = "/Users/Alexander/sparkdata/"
	        dir_list = [ f for f in listdir(path)]

	        # f_list is a list files with data
	        f_list = []
	        for n in xrange(0,len(dir_list)):
	            # list
	            for fil in listdir(path + dir_list[n]):
	                if "." not in fil and "_" not in fil :
	                    f_list.append(path + dir_list[n]+ "/" + fil)

	        # go through each file and insert each data unit individually 
	        for File in f_list:
	            for datum in File:
	                f = open(File, 'r')
	                data = f.readlines()
	                for n in xrange(0, len(data)):
	                    insert_data_hbase(eval(data[n])[0], eval(data[n])[1])

	        # empty directory once insert is complete
	        # there must be a delay between load and deleting the path
	        # I am very likely deleting data that is not being put into HBase
	        shutil.rmtree(path)

	    except OSError:
	        print "empty directory - sleep for 10 seconds"
	        time.sleep(10)



