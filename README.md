# Less-Fake-More-Good-News-Classification
Training an algorithm on hand-labeled fake news dataset with the goal of flipping the script. Rather than focus on the 'fake', we have an opportunity to highlight the good.

# Business Problem
Misinformation and disinformation is an increasing consideration in the public sphere. For the past 5 years, we've seen governments, news agencies, and social media companies grapple with misleading news and social posts. As we enter 2021, it seems more important than ever to find a consensus on information - what is true and factual. 

Social media is ingrained in societies across the world - billions of people turn to their Facebook groups, Twitter "subject-matter" experts, and other digital social outlets for news that shapes their view on the world. And with the potential virality of one false tweet, misinformation can become "factual" in the eyes of millions; a 21st-century threat that we need to grapple with. 

One common method social media companies have employed is labeling information as "disputed" of "false". Another is to take down information after the fact. Both of these have an adverse effect - those who've had posts labeled or removed are made more sure that their information is true.

I've decided to try another tactic: focus on the good, show less bad. To do so, I've ran & compared various classification models to label text (both social posts and website news stories) as either "Real" or "Fake". The intention is to find a classification model that is both accurate and lightweight enough to be used quickly.

Once we have a trained classifier, we can combine it with a recommender system to rank truth and substantive content over false and inflaming. This is a key differentiation with the current norm of the attention commodity which often leads to division and outrage (as these better hold ones attention). The actual recommender system is beyond the scope of this project - we will stick to the first step of identifying the good news so that we can hype up our better angels.

# The Data
The data was sourced from Kaggle's Source Based Fake News Classification. It contains text and metadata scraped from 244 website and labeled with the BS Detector Chrome Extension by Daniel Sieradski.

Note: We will be training our classifiers on data labeled by a classification algorithm (pretty meta, I know). In an ideal situation, we would have large datasets hand-labeled by experts (or label based on something other than "real" or "fake").

<img src="https://github.com/Stenke/Less-Fake-More-Good-News-Classification/blob/main/Images/pd-data-table-example.png" width="1100" length="1600"/>

<img src="https://github.com/Stenke/Less-Fake-More-Good-News-Classification/blob/main/Images/news-text-example.png" width="11000" length="1400" />

Source: https://www.kaggle.com/ruchi798/source-based-news-classification

## Questions
The following questions will guide our analysis and modeling as we evaluate performance.

1. How to best use NLP to process text data?

2. What aglorithm(s) will perform best on text data?

3. What aglorithm(s) best balance computational costs and accuracy?

## Methods
Text data was initially explored with visualizations showing differences in real versus fake text data. Around 50 rows were removed due to NaN values leaving us with 2050 rows of text. Data was then processed through various NLP techniques including stopword removal, tokenization, vectorization using TF-IDF. There is a class imbalnced of 2/3 'fake' and 1/3 'real'. Initially, this imbalance was left as is. Later, SMOTE was explored though without any overall improvements. There was a similar experience with dimensionality reduction using TruncatedSVD. 

Analysis of frequency distributions:
<img src="https://github.com/Stenke/Less-Fake-More-Good-News-Classification/blob/main/Images/real-fake-top-words.png" width="1200" length="2000"/>

Next, Train-Test-Split was employed with a 20% test size. Our dependent variable was the labeled data columm where 1 is Real and 0 is Fake (changed using LabelEncoder). Processed text data was used for explanatory variables with 300,000+ columns. Using Sci-Kit Learn's TF-IDF Vectorizer, trigrams were created and word count limited to 100,000 - 150,000. And now we're ready for modeling...

In order to classify our text data, seven classifier models were explored:
  1. Logistic Regressoin
  2. Decision Tree
  3. Guassian Naive Bayes
  4. Random Forest
  5. Gradient Boosting
  6. XGBoost
  7. SVM - Sigmoid & Linear Kernels
  
Model performance was evaluted based on various metrics - Accuracy, Precision, Recall, F1-Score, and Average Precision. Additionally, computational speed was considered since the viability of our model in production will depend on how quickly we can run the model. In the case of our business problem, a model to help classify text so that real news could rise to the top in a recommender system, Precision seems the most important. Precision in our case means that news that we label as real is truly real (with little false positives). Validating misinformation is dangerous and worse than no information at all (shoutout to Naruto for that notion - watching it with my lil' sis over the holiday).

Logistic Regression after tuning with confusion matrix:
<img src="https://github.com/Stenke/Less-Fake-More-Good-News-Classification/blob/main/Images/log-reg-results.png" width="700" length="900"/>
<img src="https://github.com/Stenke/Less-Fake-More-Good-News-Classification/blob/main/Images/log-reg-real-cm.png" width="600" length="600"/>

A few models were chosen for GridSearchCV based on out-of-box performance. The winners were Logistic Regression, Gradient Boosting, and SVM. XGBoost was toyed with but turns out the model is smarter than my parameter tuning attempts. An example of performing GridSearch can be found below:

<img src="https://github.com/Stenke/Less-Fake-More-Good-News-Classification/blob/main/Images/Gradient-Boost-GridSearch.png" width="1600" length="2000"/>

## Findings
Our final model process consisted of the following:

1. Vanilla Model

![Model1-summary](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/vanilla-summary.png "model1-summary")

![Model1-rmse](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/vanilla-rmse.png "model1-rmse")

![Model1-QQ](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/vanilla-qq-plot.png "Model1-QQ")

Base model R-Squared may be high from remaining spurious correlation. This would explain the large error shown by our RMSE.

Additionally, the QQ plot is not normal (in the worst kind of way) with a heavy tail on the upper end. Outlier trimming to the rescue!

2. Model 2: Pricing Outliers Removed

![Model2-summary](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/outliers-summary.png "model2-summary")

![Model2-rmse](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/outliers-rmse.png "model2-rmse")

![Model2-QQ](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/outliers-qq-plot.png "Model2-QQ")

Narrowing the price range of our model reduced R-Squared but improved the accuracy by over 50%. Additionally, our QQ plot looks more normal (and cuter).

3. Model 3: Scaled Explantory Variables (Min/Max)

![Model3-summary](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/scale-summary.png "model3-summary")

![Model3-rmse](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/scale-rmse.png "model3-rmse")

4. Model 4: Find & Add Interactions

![Model4-summary](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/interactions-summary.png "model4-summary")

![Model4-rmse](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/interactions-rmse.png "model4-rmse")

5. Model 5: Polynomial Variables Added

![Model5-summary](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/poly-summary.png "model5-summary")

![Model5-rmse](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/poly-rmse.png "model5-rmse")

6. Model 6: P-Value Filtered (Stepwise Function)

![Model6-summary](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/final-summary.png "model6-summary")

![Model6-rmse](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/final-rmse.png "model6-rmse")

![Model6-report](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/final-report.png "model6-report")

![Model6-resid-plot](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/final-model-residuals-plot.png "model6-resid-plot")

Our final model has an Adjusted R-Squared of 0.760 meaning 76% of the variability in house pricing (dependent variable) can be explained by our explantory variables in our model. Additionally, we reduced the root-mean-squared-error to 77,710, a 56.52% improvement. This means we can more accurately predict the housing price with less error on either end. The test prediction model was within 1.10% of the training model. Additionally, our cross-validated R-Squared is only 1.0% different from our training model. This is shows that the generalization of our model in the wild is promising. Of course, we'll never know until we try it!


## Conclusion
We are focusing on homes in the $200,000 - $790,000 price range initially. This is both what our model is best suited for and aligns with opportunities we see in the data. Using the most impactful variables based on coefficients, we found specific opportunities especially based on cities. After looking at additional data provided by King County, we noticed that house pricing seemed to correlate with kindergarten readiness and enrollment. We suspect this extends to other school metrics.

Below is a graph of Percentage Kindergarten Readiness of select school districts in King County as referenced earlier.

![kc-kindergarten-graph](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/kindergarten-graph.png "kindergarten-readiness")

With this in mind, we feel especially equipped for helping young families with well-paying jobs find their first home in Seattle and its suburbs. We found opportunities for customers who are looking for housing in a great school district for younger children that is still affordable. One example of this is Lake Washington School District, which includes the neighboring cities of Kirkland and Redmond. In our model, we see that Kirkland has a large positive coefficient, which is to be expected as it is known to be a wealthy suburb of Seattle. Right next door is Redmond that our model found a negative impact on pricing, meaning we can find affordable housing in a great school district and close access to similar ammenities. At Rest, we feel especially equipped for finding these opportunities for our young and aspiring families.

![kc-redmond-kirkland](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/KingCounty-Kirkland%2BRedmond.png "redmond-kirkland")

