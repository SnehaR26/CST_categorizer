# CST_categorizer
A project which aims to sort a customer support ticket according to the relevant category

---
#### Inspiration:
This project was inspired by my frustration of logging an online complaint with a service provider where I had to scroll through many complaint categories and see which one fit best. Some of these sounded too similar and overall the whole thing took much of my time. I wanted to build a feature where a customer can just describe the complaint and have the system automatically categorize and log it.

#### Challenges
This sounds similar to a classification problem but here the input is a text description. We need to convert them into embeddings and then attempt to classify. The solution should capture context from embeddings which becomes difficult in higher dimensions.

#### Dataset
We will use this customer support ticket dataset from kaggle URL(https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) 

#### EDA
We can see the column 'Ticket Description' captures the textual data pertaining the customer issue. We also have 'Ticket Subject' which provides the class label for type of ticket. 
There are 16 categories each with approx 500 tickets 
|           Ticket Subject | Count |
|-------------------------:|------:|
|      Refund request      |   576 |
|       Software bug       |   574 |
|   Product compatibility  |   567 |
|     Delivery problem     |   561 |
|      Hardware issue      |   547 |
|       Battery life       |   542 |
|      Network problem     |   539 |
|   Installation support   |   530 |
|       Product setup      |   529 |
|       Payment issue      |   526 |
|  Product recommendation  |   517 |
|      Account access      |   509 |
| Peripheral compatibility |   496 |
|         Data loss        |   491 |
|   Cancellation request   |   487 |
|       Display issue      |   478 |

Other observations are captured in <>

#### Approach 1
I initially sought to convert these ticket description texts into embeddings using 4 approaches namely 
- TF-IDF
- GLOVE
- BERT CLS
- Sentence Transformers

After generating embeddings I tried to use clustering, an unsupervised learning approach to find the clusters in data. I used K means for clustering
##### Results
We dont see any segmentable clusters in any of these techniques. This could be caused from poor embeddings or the descriptions being full of noise. 
![image](https://github.com/user-attachments/assets/e8d5280e-d1d8-40d4-9cc5-183472a3dc6e)
