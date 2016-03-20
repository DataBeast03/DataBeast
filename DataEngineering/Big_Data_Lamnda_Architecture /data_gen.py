import numpy 
import pandas
import time
import uuid
import random
from os import listdir
from os.path import isfile, join
from nltk.corpus import names
import json
import random



def get_itemName_itemReview():
    path = "/Users/Alexander/Downloads/MicropinionDataset_09-17-2012/pre-processed"
    file_list = [ f for f in listdir(path) if isfile(join(path,f)) ]

    reviews = []
    item_name = []
     
    random.shuffle(file_list)
    N = len(file_list)
    for n in xrange(N):
        # used to concatnate lines in each review
        review_temp = ""
        # get item name
        item_name.append(file_list[n].rstrip(".data"))
        
        f = open(path + "/" + file_list[n], 'r')
        raw_review = f.readlines()
        
        for line in raw_review:
            review_temp += line
        
        reviews.append(review_temp)
    
    return item_name, reviews

def gen_itemProperties_ReviewEdge_PurchaseEdge_SellerEdge(userID_list,itemID_list, name_list,review_list):
    '''
    For every 10 item names generated, 1 item review will be generated.
    There is a 50 percent chance that an item will get a rating.
    An item may be tagged multiple times with different names.
    Dublicates will be filtered in data normalization phase.
    '''        
    # pair in tuples: (file_name,file_review)
    name_review = [(name,review) for name, review in zip(name_list,review_list)]
    
    names_generated = 0 
    # keep record of items have been assinged names
    instaniated_items = []
    

    get_time = adj_time()

    while True:
        # get random ith_name_review index
        item_i = numpy.random.choice(len(name_review))
        # get ith_name value
        item_name , _ = name_review[item_i]
        
        # get random item id
        item_id = numpy.random.choice(itemID_list)
        
        # store (item_id, ith_name_review_index)_ith
        instaniated_items.append((item_id,item_i))
        
        get_time = adj_time()
        # generate item name
        yield {"pedigree": {"true_as_of_secs": get_time.next()},
                   "dataunit": {"item_property": {"item_id": {"item_number": item_id},
                                                 "property": {"item_name"  : item_name}}}}
        names_generated += 1
        
        if names_generated % 10 == 0:
            "Ratings go to items that already have a name and in 'instaniated_items'  "
            # get random user
            buyer_id = numpy.random.choice(userID_list)
            seller_id = numpy.random.choice(userID_list)
            # get random j_th item
            item_jth =  numpy.random.choice(len(instaniated_items))
            # get (item_id, ith_name_review_index)
            item_id, _ = instaniated_items[item_jth]
            # pick ith_review - making sure that the correct item review is picked
            #_ , review = name_review[item_ith]
            
            # generate purchase                                   
            yield {"pedigree": {"true_as_of_secs": get_time.next()},
                   "dataunit": {"purchase_edge": {"userID": {"user_number": buyer_id},
                                                 "item_id": {"item_number": item_id},
                                          "purchase_time" : get_time.next()}}}
                                       
            
            random_int = random.randint(1,11)
            if random_int < 3:
                # rating is between 1 and 5 (inclusive)
                rating = numpy.random.choice(numpy.arange(1,6))
                # generate Item Rating                       
                yield  {"pedigree": {"true_as_of_secs": get_time.next()},
                       "dataunit": {"review_edge":{"userID"  : {"user_number": buyer_id},
                                                   "item_id" : {"item_number": item_id},
                                                    "review" :   "null",
                                                     "rating": rating}}}
            
            continue 


def adj_time():
    hour = 60 * 60
    day = 24 * hour
    week = 7 * day
    while True:
        frac = numpy.random.random_sample() # value between 0 and 1
        yield int(time.time() + frac * week)



def get_gender():
    if numpy.random.random() > 0.51:
        return "MALE"
    else:
        return "FEMALE"

def gen_userProperties(user_list):
    
    Names = names.words()

    get_time = adj_time()

    def get_city_state_bank():
        df = pandas.read_csv("/Users/Alexander/Downloads/AdWords API Location Criteria 2015-05-29.csv")
        city_state = []
        for location in df["Canonical Name"].values:
            if location.split(",")[-1] == "United States":
                try:
                    city_state.append((location.split(",")[0],location.split(",")[-2]))
                except: IndexError
        return city_state



    city_state_bank = get_city_state_bank()
    city, state = random.sample(city_state_bank,1)[0]
    while True:
        user_id = numpy.random.choice(user_list)
        random_int = numpy.random.choice(numpy.arange(1,11))
        
        if (random_int <= 6):
            first = numpy.random.choice(names.words())
            last = numpy.random.choice(names.words())
            
            yield {'pedigree': {'true_as_of_secs': get_time.next()},
            'dataunit': {'user_property': {'userID': {'user_number': user_id},
                            'property': {'user_name': {"name": first + " " + last}}
                                          }                                          
                       }

                  }

        elif (random_int > 6 and random_int <= 8):
            yield {"pedigree": {'true_as_of_secs': get_time.next()},
                   "dataunit": {"user_property": {"userID": {"user_number": user_id},
                                                "property": {"gender": get_gender()}
                                                 }
                               }
                  }
        else:
            # generate random city and state
            city, state = random.sample(city_state_bank,1)[0]
            yield {"pedigree": {'true_as_of_secs': get_time.next()},
                   "dataunit": {"user_property": {"userID": {"user_number": user_id},
                                                "property": {"location": {"city" :  city,
                                                                          "state": state,
                                                                         "country": "USA"}}
                                                  }
                                }
                   }


def gen_ID():
    while True:
        yield str(uuid.uuid1().get_time())+ time.strftime("_%Y_%H_%M", time.gmtime())

def fix_text(reviews):
    return [  unicode(review, errors='ignore')   for review in reviews]

def save_bank_to_file(UserID_bank):
    # save user ID to file
    with open('UserID_bank.txt','w') as f: 
        try:
            f.write(json.dumps(UserID_bank))
        finally:
            f.close()


def get_datum(save_to_file = False, reuse_user_ids = True):
    N_users = 1000
    N_items = 500
    item_names, item_reviews_temps = get_itemName_itemReview()
    item_reviews = fix_text(item_reviews_temps)
    gen_id = gen_ID()
    ItemID_bank = [gen_id.next() for _ in xrange(0,N_items)]

    # only want to generate new ids once
    # then reuse those ids 
    if reuse_user_ids == False:
        UserID_bank = [gen_id.next() for _ in xrange(0,N_users)]
    else: 
        with open('UserID_bank.txt','r') as f:
            UserID_bank = json.loads(f.readlines()[0])

    if save_to_file == True:
        save_bank_to_file(UserID_bank)

    gen_item_prop_review_edge = gen_itemProperties_ReviewEdge_PurchaseEdge_SellerEdge(UserID_bank,\
                                                                                      ItemID_bank,\
                                                                                       item_names,\
                                                                                       item_reviews)

    gen_userprop = gen_userProperties(UserID_bank)
    while True:
           if random.randint(1,2) == 1:
               yield gen_item_prop_review_edge.next()
           else:                                                                        
               yield gen_userprop.next()



