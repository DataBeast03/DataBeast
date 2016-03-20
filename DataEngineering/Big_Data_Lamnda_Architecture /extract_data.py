def get_purchases(datum):
    item = datum['dataunit']['purchase_edge']['item_id']['item_number']
    user = datum['dataunit']['purchase_edge']['userID']['user_number']
    true_as_of_secs = datum['pedigree']['true_as_of_secs']
    return user, item,true_as_of_secs

def get_user_gender(gender_datum):
    
    gender  = gender_datum['dataunit']['user_property']['property']['gender']
    user_id = gender_datum['dataunit']['user_property']['userID']['user_number']
    ts      = gender_datum['pedigree']['true_as_of_secs']
    
    return user_id, gender, ts

def get_user_location(location_datum):
    
    city    = location_datum['dataunit']['user_property']['property']['location']['city']
    state   = location_datum['dataunit']['user_property']['property']['location']['state']
    country = location_datum['dataunit']['user_property']['property']['location']['country']
    user_id = location_datum['dataunit']['user_property']['userID']['user_number']
    ts      = location_datum['pedigree']['true_as_of_secs']
    
    return user_id, city,state,country, ts

def get_user_name(name_datum):
    
    name    = name_datum['dataunit']['user_property']['property']['user_name']['name']
    user_id = name_datum['dataunit']['user_property']['userID']['user_number']
    ts      = name_datum['pedigree']['true_as_of_secs']
    
#     return user_id, name, ts
    return  user_id, name, ts

def get_itemName(item_datum):
    ID     = item_datum['dataunit']['item_property']['item_id']['item_number']
    name    = item_datum['dataunit']['item_property']['property']['item_name']
    ts      = item_datum['pedigree']['true_as_of_secs']
    return ID, name, ts

def get_review_rating(review_datum):
    user_id = review_datum['dataunit']['review_edge']['userID']['user_number']
    item_id = review_datum['dataunit']['review_edge']['item_id']['item_number']
    rating  = review_datum['dataunit']['review_edge']['rating']
    review  = review_datum['dataunit']['review_edge']['review']
    ts      = review_datum['pedigree']['true_as_of_secs']
    return user_id, item_id, rating, ts, review    