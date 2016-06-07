# Lambda Architecture for Big Data

Lambda Architecture is a data pipelined desinged with big data in mind. Due to the velcoity, varitety, and volume of big data,
it is necessary to divide the pipeline into layers. 

![The master dataset in the Lambda Architecture serves as the source of truth for your Big Data system. 
Errors at the serving and speed layers can be corrected, but corruption of the master dataset is irreparable]
(https://s3-us-west-2.amazonaws.com/dsci6007/assets/fig2-1.png)

## Batch Layer 

The master data set is structed into batch views. These batch views are relatively small, managable sizes that any query 
sent to one or more batch views will be served in a reasonable amount of time. Batch views are regenerated periodically. 
How often batch views are regenerated will depend on the velocity and volume of the data being collected. The batch layer 
can be thought of as the backend of the pipeline

## Serving Layer
Whenever a user sends a query, the serving layer will reference the relavent batch view or batch views, aggragate data, and 
return the results for the user's query with low latency. The serving layer can be thought of as the front end of the pipeline. 

## Speed Layer 
Due to the high latency of the batch layer, the results of a user's query can be missing data that was collect while the 
batch layer was regenerating batch views. The speed layer enables the results for a user's query to be up-to-date 
with the latest data as it is collected. The speed layer operates indepdendent of the batch and serving layers. 
