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

![data](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/Screen%20Shot%202020-11-29%20at%206.38.05%20PM.png "initial data")

Source: https://www.kaggle.com/ruchi798/source-based-news-classification

## Questions
The following questions will guide our analysis and modeling as we look for opportunities for our business.

1. What price range are catering towards?

2. Are there specific locations where we should focus?

3. What features are most important for predicting home prices?

4. Any additional opportunities uncovered in the analysis?


## Methods
Data obtained from King Country housing dataset. Data was scrubbed and cleaned to remove outliers and null values. Null values were filled either with 0's or the median values depending on the situation. Additionally, we binned certain features like bathrooms to reduce inputs and transformed other features into binary columns like renovated (either yes or no instead of years renovated). We then looked at a scatter matrix to get an idea of which explanatory variables may be useful.

![Scatter-Matrix](https://github.com/Stenke/King-County-Housing-Regression-Analysis/blob/main/Figures/Scatter-Matrix.png "EDA-Scatter-Matrix")

For our various models, dummy variables were created for categorical data. We also replaced zipcodes with their respective cities in King County. In order to meet assumptions for our models, we removed outliers by limiting pricing data to 2 standard deviations (95%).

Below is a snippet of the bathroom binning code.

![Bathroom-bin-code](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/bathroom-bins.png "Bathroom-bin-code")

![Bathroom-bin-plot](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/bathroom-bin.png "Bathroom-bin-plot")

And here is the library and code used to determine cities of King County from zipcodes.

![Zipcodes](https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/zipcode-code.png "Zipcode-to-city")

After assuring regression assumptions, we used Scikit-Learn and Statsmodels libraries to create a baseline model for predicting housing prices. Using train-test-split, we iterated on the model to remove collinearity, extraneous variables, and created features all while aiming for an improved R-squared value and reduced RMSE. QQ plots were used to determine normality in our models. Finally, we performed cross-validation to ensure our model meets our expecation for generalization.

Here we check collinear variables seen as the tuples below.

<img src="https://github.com/Stenke/Seattle-Housing-Regression-Analysis/blob/main/Figures/collinearity.png" width="200" length="300" />

Additional processing code and graphs can be found in the EDA notebook.

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

