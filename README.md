# Opinion-Spammers-Detect-users-with-too-many-negative-reviews-for-movies
1. Introduction and Problem Description : 

Authenticity plays a substantial role in online purchases. The product reviews posted in the websites acts as a major deciding factor for purchasing a particular product. Off late, consumers are facing bigger issues with deceiving reviews posted in the websites. Such fake reviews contribute towards creating a phantom feedback for the products. The objective of this project is to detect the users with too many negative or fake reviews for the movies dataset collected from Amazon.com. We employ the textual information contained in the reviews along with other users rating (helpfulness) for the particular review posted, the time frame which the user posted the review is also taken into consideration. We calculate the sentiment score and assign labels to each of the reviews. The major challenge in this problem is that there is no Ground Truth available and also it is impossible to accurately generate one for the dataset.


2. Related work : 

There are several existing works addressing this problem. Fakespot is a website which gives the authenticity score for any reviews you enter. Based on several factors such as the similarity of the reviews to other reviews, helpfulness score of the review and other user characteristics are considered for obtaining the authenticity score. There are several other techniques used to solve this problem but all of them face the same initial challenge of the unavailability of Ground Truth. In few approaches the reviews are manually scanned and labelled. In this project we perform Sentiment Analysis on the review text and utilize the results to label the reviews. 



3.Dataset description : 

We are provided with the Movies.txt file. The dataset consists of movie reviews from Amazon. The data ranges from a period of August 1997 to October 2012, with an approximate amount of 8 billion reviews. The data includes information pertaining to the users, movies and their ratings, how many users found this review to be helpful, and actual reviews in plain text format. The data is in a semi-structured format and the details pertaining to a single review occupy 8 lines. Each attribute describing a review are given in a key:value format on separate lines. 



Dataset statistics:

Number of reviews			7,911,684
Number of users			889,176
Number of products			253,059
Users with > 50 reviews		16,341
Median no. of words per review	101
Timespan				Aug 1997 - Oct 2012

 Features:
The total number of features that will be used for prediction: 3
The total number of instances: 7,911,684
                                                        	

Attributes in the dataset are:
productId : movie id 
userId : user id
helpfulness : fraction of users who found the review helpful
score : rating of the product
time : time of the review (unix time)
summary : review summary
text : text of the review 

We would be using the following attributes as features:
userId
words extracted from the text
sentiment_score


4. Pre-processing technique : 

The data provided is not in any regular format . This raw data needs to be converted to a useful format for further use. For this purpose, we have used Python and grouped data belonging to a particular review into rows and converted it to csv format. This converted data then had special characters which were removed as this affects our further techniques. The transformation of the dataset from a text file to csv file was done by reading each line of the text file and splitting based on the ‘:’ character. Since each review contains 8 lines of data describing it, we group the information present in these 8 lines into a single row in the csv file. This data is loaded into a dataframe and the helpfulness factor which is a fraction is divided into two separate columns (numerator, denominator) for ease of access. The data had some null values. All these are removed by  using the “ DataFrame.na.drop() ” command in scala. 
				         Screenshot 2: Preprocessed Dataset


5. Proposed solution : 

AFINN Sentiment Lexicon is a file in which each word is given a sentiment score depending upon the strength of the sentiment expressed when utilizing that word. We downloaded this file from online sources and utilized it to calculate the sentiment of a review given by the user to a movie. Although, the data consisted of the review score column which gives us the score that a user has given to a movie, we thought of taking sentiment of the review text into consideration because review score is not a normalized quantity. This is because every person’s level of score for a given sentiment might be different. For example, Person A and Person B might think that a movie is bad. But the review score they give may be 2 /5 and 3 /5 respectively. Though both the reviewers have same opinion, their scores differ. Also, it depends on the perspective of the user reading it. The person viewing the rating might have a different scale of score compared to the person who had given that rating. For normality and better predictions, we have considered the sentiment of the review which is calculated in a more uniform manner by utilizing the same sentiment score for a particular word across all reviews. Each review is broken down into words and the sentiment of each word is calculated and is aggregated to get the overall sentiment of a review. As stop words mislead the model, we have removed the stop words. 

We have then calculated the average of the sentiment score for all the reviews. This score is then compared with the average score of the user. Users with sentiment score greater than the average score are assigned a label of 1 and users with sentiment score less than the average but within two standard deviations are also assigned a label of 1. Users below that are assigned a label of 0 signifying negative reviewers. 

The entire data is split into 80% for the training data and 20% for the test data. Then , this training data is sent to a ML pipeline whose steps include tokenizing and removing stop words, calculating features using HashingTF and IDF followed by a ML model. We have used this data to feed into two models namely Logistic regression and Random Forest. This is fitted to the training data and used on test data to make predictions. Binary Classification evaluator is used for evaluation of the results. Area under ROC curve is used for analysis of performance. ParamGridBuilder is used for utilizing the best set of parameters. We also use 5 fold Cross Validation to test the models as we didn’t any separate test dataset. The results are then compared for fitted and the non-tuned models.

The Machine Learning model would be trained to identify the words associated with different sentiment levels. Then the model classifies the unseen reviews based on the words occurring in it about which the model has some prior knowledge. 

Random Forest is an ensemble technique where the base classifiers are Decision Trees. In Random Forest, the idea is to create Decision Tree models from bootstrap samples and to limit the splitting criteria of these trees to a random sample of m features where the value of m is less than the total number of features available in the training data (that is only a subset of all the features is used for each decision tree). The output which is produced by majority of these decision tree models is considered as the output of the Random Forest Classifier.
Pseudocode for Random Forest:
Consider a dataset D with N instances and M features

Training:

for i=1 to number of estimators
create a bootstrap sample of size N from the original sample construct tree as below:
        	generate a random sample of m features (m<M)
           use the features from the above sample for splitting the nodes

Testing:

for each estimator in the forest move along the decision tree depending on the values in the test point
output the class produced by majority of the decision trees


8. Conclusion : 

The overall aim is to find the users with too many negative reviews. We have used sentiment analysis to find out the sentiment of the review and used this data for predictions using ML models like logistic regression and random forest. These models gives predictions regarding  whether a review is a case of negative opinion spamming or not. We also evaluate the strength of the Machine Learning models in predicting the class labels using AuROC.  
