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

Other observations are captured in [EDA](EDA_CST.ipynb)
The below approaches can be found in [Ticket_topic](Ticket_topic.ipynb)
#### Approach 1
I initially sought to convert these ticket description texts into embeddings using 4 approaches namely 
- TF-IDF
- GLOVE
- BERT CLS
- Sentence Transformers

After generating embeddings I tried to use clustering, an unsupervised learning approach to find the clusters in data. I used K means for clustering
##### Results
We don't see any segmentable clusters in any of these techniques. Even metrics such as Adjusted Rand Index (ARI), Normalized Mutual Information (NMI) and, Fowlkes-Mallows Index (FMI) indicate poor performnece. This could be caused from poor embeddings or the descriptions being full of noise. 
![image](https://github.com/user-attachments/assets/2cdabd29-76f6-4bd0-8fa8-95969abdde46)


#### Approach 2

Next I tried training a model to predict label. After cleaning and preprocessing, text data was converted into numerical representations using 
- TF-IDF Vectorizer (Term frequency-inverse document frequency
- Sentence Transformers (BERT-based): Contextual embeddings
- GloVe Embeddings: Global vectors for word representation

Then I applied different combinations of machine learning models:
- TF-IDF + Naive Bayes
- TF-IDF + Logistic Regression
- TF-IDF + Random Forest
- GloVe + Random Forest
- Sentence Transformers + Logistic Regression

Also tried Deep Learning  for improved performance:
- BERT for Text Classification
- LSTM with GloVe embeddings
  
##### Results
Even with deep learning approaches, we got poor results. Accuracy of around 6-7% and almost all metrics (precision, recall, F1-score) at 0 for most classes. 
It could have been caused by Data issues such as Noisy data, underfitted model, dataset being too small etc.

#### Approach 3
I approached the problem from an unsupervised point of view. Instead of using predefined topics, which are typical across various industries, we can try defining customized topics based on a company's dataset. BERTTopic addresses this problem. 
It uses BERT or other transformer models to create dense vector representations (embeddings) of each item in the corpus.
The embeddings are then clustered, using HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).
Each cluster represents a potential topic.
Advantages here are that we don't have to specify the number of clusters in advance and the number of clusters can be reduced after model fitting.

##### Results 
I got some satisfactory results with this approach when I tested it for a new ticket
![image](https://github.com/user-attachments/assets/1f5cd8d3-e485-4b0e-9e6a-dc6c43858243)
