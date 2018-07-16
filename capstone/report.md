# Machine Learning Engineer Nanodegree
## Predict College Earning Potential
Aloysius Joseph  
July 15, 2018

## I. Definition

### Project Overview

   I propose to create a model for prediction for college selection based on earning potential. Students and parents have a tough time determining which colleges to apply. There are a lot of factors to consider and lots of conflicting information. Also there is lots of data available as well as lots of variables involved. But in general apart from SAT score and GPA that are used mainly for the admission process, several factors like University admission rate, public/private type of university etc., need to be considered.

Dataset: https://collegescorecard.ed.gov/data/Most-Recent-Cohorts-All-Data-Elements.csv

Documentation: https://collegescorecard.ed.gov/assets/FullDataDocumentation.pdf

Data Dictionary: https://collegescorecard.ed.gov/data/CollegeScorecardDataDictionary.xlsx


### Problem Statement

   Predict some of the major factors to be considered by a student when applying to Universities and help in the process of selecting Universities to apply.

   This is a classification problem. If the student is hoping to earn at least $50,000 10 years after graduation, which universities might he plan on applying (without taking the degree major into consideration). Which of these factors available about Universities will matter most: SAT scores, size (number of students), spending per student by university, type (public/private), cost for students, rate of admission, rate of completion and rate of retention. I plan to use Ensemble methods like Ada Bost, Random Forest and Gradient Boost and select the one providing better accuracy.


### Metrics

   Accuracy measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions.

   Precision is a ratio of true positives (students classified as earning >=50k, and who are actually earning that much) to all positives (all students classified as earning >=50k, irrespective of whether that was the correct classification).

    [True Positives/(True Positives + False Positives)]

   Recall(sensitivity) is a ratio of true positives(students classified as earning >=50k, and who are actually earning that much) to all the students who were actually earning >=50k.

    [True Positives/(True Positives + False Negatives)]

   For classification problems with distributions like in our case, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score. The weight β can be increased if we want to have more emphasis on precision.

    [Fβ=(1+β2)⋅precision⋅recall/(β2⋅precision)+recall]


## II. Analysis

### Data Exploration

   The dataset is provided bu the US Department of Education (https://collegescorecard.ed.gov/). It is provided as a CSV file. 
The dataset consists of approximately 7593 data points, with each datapoint having 1825 features. 

   But for the purpose of this project 9 features have been selected with 1 target variable, as these seem to be more appropriate for the problem at hand.

**Features**
- `CONTROL`: integer : Collge type (1-Public/2&3-Private)
- `ADM_RATE`: float : Admission Rate
- `SATVRMID`: float: Average SAT score in English
- `SATMTMID`: float : Average SAT score in Math
- `COSTT4_A`: integer : Average cost to complete education
- `C150_4`: float: The degree completion rate
- `RET_FT4_POOLED`: float: Student retention rate
- `NUM4_PUB`: integer : Total number of enrolled students (indicates size)
- `INEXPFTE`: integer: Instructional expenditure per student


**Target Variable**
- `MN_EARN_WNE_INC2_P10`: float: Mean earnings of students 10 years after entry (<=50K, >50K)


#### Data Sample:

<div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>UNITID</th>
          <th>OPEID</th>
          <th>OPEID6</th>
          <th>INSTNM</th>
          <th>CITY</th>
          <th>STABBR</th>
          <th>ZIP</th>
          <th>ACCREDAGENCY</th>
          <th>INSTURL</th>
          <th>NPCURL</th>
          <th>...</th>
          <th>RET_FT4_POOLED_SUPP</th>
          <th>RET_FTL4_POOLED_SUPP</th>
          <th>RET_PT4_POOLED_SUPP</th>
          <th>RET_PTL4_POOLED_SUPP</th>
          <th>TRANS_4_POOLED</th>
          <th>TRANS_L4_POOLED</th>
          <th>DTRANS_4_POOLED</th>
          <th>DTRANS_L4_POOLED</th>
          <th>TRANS_4_POOLED_SUPP</th>
          <th>TRANS_L4_POOLED_SUPP</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>100654</td>
          <td>100200</td>
          <td>1002</td>
          <td>Alabama A &amp; M University</td>
          <td>Normal</td>
          <td>AL</td>
          <td>35762</td>
          <td>Southern Association of Colleges and Schools C...</td>
          <td>www.aamu.edu/</td>
          <td>www2.aamu.edu/scripts/netpricecalc/npcalc.htm</td>
          <td>...</td>
          <td>0.61638362831858</td>
          <td>NaN</td>
          <td>0.41664791666666</td>
          <td>NaN</td>
          <td>0.200384</td>
          <td>NaN</td>
          <td>2086.0</td>
          <td>NaN</td>
          <td>0.20038350910834</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>100663</td>
          <td>105200</td>
          <td>1052</td>
          <td>University of Alabama at Birmingham</td>
          <td>Birmingham</td>
          <td>AL</td>
          <td>35294-0110</td>
          <td>Southern Association of Colleges and Schools C...</td>
          <td>www.uab.edu</td>
          <td>uab.studentaidcalculator.com/survey.aspx</td>
          <td>...</td>
          <td>0.80765744125326</td>
          <td>NaN</td>
          <td>0.58823529411764</td>
          <td>NaN</td>
          <td>0.241619</td>
          <td>NaN</td>
          <td>2740.0</td>
          <td>NaN</td>
          <td>0.24161927007299</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>100690</td>
          <td>2503400</td>
          <td>25034</td>
          <td>Amridge University</td>
          <td>Montgomery</td>
          <td>AL</td>
          <td>36117-3553</td>
          <td>Southern Association of Colleges and Schools C...</td>
          <td>www.amridgeuniversity.edu</td>
          <td>www2.amridgeuniversity.edu:9091/</td>
          <td>...</td>
          <td>PrivacySuppressed</td>
          <td>NaN</td>
          <td>PrivacySuppressed</td>
          <td>NaN</td>
          <td>0.111111</td>
          <td>NaN</td>
          <td>18.0</td>
          <td>NaN</td>
          <td>PrivacySuppressed</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>100706</td>
          <td>105500</td>
          <td>1055</td>
          <td>University of Alabama in Huntsville</td>
          <td>Huntsville</td>
          <td>AL</td>
          <td>35899</td>
          <td>Southern Association of Colleges and Schools C...</td>
          <td>www.uah.edu</td>
          <td>finaid.uah.edu/</td>
          <td>...</td>
          <td>0.78698579881656</td>
          <td>NaN</td>
          <td>0.50876842105263</td>
          <td>NaN</td>
          <td>0.332677</td>
          <td>NaN</td>
          <td>1539.0</td>
          <td>NaN</td>
          <td>0.33267738791423</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>100724</td>
          <td>100500</td>
          <td>1005</td>
          <td>Alabama State University</td>
          <td>Montgomery</td>
          <td>AL</td>
          <td>36104-0271</td>
          <td>Southern Association of Colleges and Schools C...</td>
          <td>www.alasu.edu</td>
          <td>www.alasu.edu/cost-aid/forms/calculator/index....</td>
          <td>...</td>
          <td>0.58470804331013</td>
          <td>NaN</td>
          <td>0.43181818181818</td>
          <td>NaN</td>
          <td>0.000000</td>
          <td>NaN</td>
          <td>2539.0</td>
          <td>NaN</td>
          <td>0</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
</div>



#### Data Statistics for selected features and target:

<div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Earning</th>
          <th>NoStudents</th>
          <th>UnivSpending</th>
          <th>UnivType</th>
          <th>SATMath</th>
          <th>SATRead</th>
          <th>StudentCost</th>
          <th>AdmissionRate</th>
          <th>Completionrate</th>
          <th>RetentionRate</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>448.000000</td>
          <td>448.000000</td>
          <td>448.000000</td>
          <td>448.0</td>
          <td>448.000000</td>
          <td>448.000000</td>
          <td>448.000000</td>
          <td>448.000000</td>
          <td>448.000000</td>
          <td>448.000000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>47238.392857</td>
          <td>1038.589286</td>
          <td>9676.296875</td>
          <td>1.0</td>
          <td>512.169643</td>
          <td>524.897321</td>
          <td>21166.167411</td>
          <td>0.682814</td>
          <td>0.511222</td>
          <td>0.768067</td>
        </tr>
        <tr>
          <th>std</th>
          <td>8848.756487</td>
          <td>805.681157</td>
          <td>4378.046492</td>
          <td>0.0</td>
          <td>55.772049</td>
          <td>62.385990</td>
          <td>4059.443911</td>
          <td>0.168965</td>
          <td>0.170033</td>
          <td>0.096141</td>
        </tr>
        <tr>
          <th>min</th>
          <td>29800.000000</td>
          <td>54.000000</td>
          <td>3121.000000</td>
          <td>1.0</td>
          <td>370.000000</td>
          <td>380.000000</td>
          <td>12151.000000</td>
          <td>0.168836</td>
          <td>0.108900</td>
          <td>0.488141</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>41100.000000</td>
          <td>387.500000</td>
          <td>7039.750000</td>
          <td>1.0</td>
          <td>475.000000</td>
          <td>485.000000</td>
          <td>18452.500000</td>
          <td>0.582301</td>
          <td>0.397175</td>
          <td>0.703396</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>46200.000000</td>
          <td>828.500000</td>
          <td>8591.000000</td>
          <td>1.0</td>
          <td>505.000000</td>
          <td>515.000000</td>
          <td>20949.500000</td>
          <td>0.690453</td>
          <td>0.493250</td>
          <td>0.766676</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>51300.000000</td>
          <td>1476.750000</td>
          <td>10817.500000</td>
          <td>1.0</td>
          <td>545.250000</td>
          <td>555.000000</td>
          <td>23340.000000</td>
          <td>0.805161</td>
          <td>0.632250</td>
          <td>0.840963</td>
        </tr>
        <tr>
          <th>max</th>
          <td>95900.000000</td>
          <td>3975.000000</td>
          <td>43996.000000</td>
          <td>1.0</td>
          <td>680.000000</td>
          <td>745.000000</td>
          <td>34496.000000</td>
          <td>1.000000</td>
          <td>0.933000</td>
          <td>0.972121</td>
        </tr>
      </tbody>
    </table>
</div>



#### Data Sample for selected features and target:

<div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Earning</th>
          <th>NoStudents</th>
          <th>UnivSpending</th>
          <th>UnivType</th>
          <th>SATMath</th>
          <th>SATRead</th>
          <th>StudentCost</th>
          <th>AdmissionRate</th>
          <th>Completionrate</th>
          <th>RetentionRate</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>35500.0</td>
          <td>743.0</td>
          <td>7941.0</td>
          <td>1</td>
          <td>427.0</td>
          <td>420.0</td>
          <td>20809.0</td>
          <td>0.653841</td>
          <td>0.3081</td>
          <td>0.616384</td>
        </tr>
        <tr>
          <th>1</th>
          <td>45900.0</td>
          <td>955.0</td>
          <td>17548.0</td>
          <td>1</td>
          <td>575.0</td>
          <td>594.0</td>
          <td>22232.0</td>
          <td>0.604275</td>
          <td>0.5462</td>
          <td>0.807657</td>
        </tr>
        <tr>
          <th>3</th>
          <td>53400.0</td>
          <td>331.0</td>
          <td>10619.0</td>
          <td>1</td>
          <td>585.0</td>
          <td>615.0</td>
          <td>20999.0</td>
          <td>0.811971</td>
          <td>0.4935</td>
          <td>0.786986</td>
        </tr>
        <tr>
          <th>4</th>
          <td>30700.0</td>
          <td>570.0</td>
          <td>7742.0</td>
          <td>1</td>
          <td>410.0</td>
          <td>410.0</td>
          <td>18100.0</td>
          <td>0.463858</td>
          <td>0.2696</td>
          <td>0.584708</td>
        </tr>
        <tr>
          <th>5</th>
          <td>50100.0</td>
          <td>1282.0</td>
          <td>10312.0</td>
          <td>1</td>
          <td>545.0</td>
          <td>550.0</td>
          <td>27205.0</td>
          <td>0.535867</td>
          <td>0.6709</td>
          <td>0.865822</td>
        </tr>
      </tbody>
    </table>
</div>    
    
#### Final DataSet details:
   Some of the data records did not have information for Earnings from Universities as it was privacy protedted. 
Hence had to pre-cleanup to remove those records.

    Total number of records: 448
    Individuals making more than $50,000: 150
    Individuals making at most $50,000: 298
    Percentage of individuals making more than $50,000: 33.48%


### Exploratory Visualization

The various distributions of the Features are displayed below:

#### Observations:
   The distributions for number of students does not have a normal distribution and skewed to left, as there are large universities and small private colleges. The disctribution for niversity spending on students is highly skewed to the left, as few famous universities have huge endowments while most others do not have that type of funding to spend on students.
Hence we may have to do log-transformation for these features so they do not negatively affect the performance of a learning algorithm.

<img src="images/output_11_1.png" />

   The distribution of other featues namely student cost, admission rate, completion rate, retention rate and SAT scores generally have somewhat normal distributions which might be good enough for this prediction.

<img src="images/output_11_2.png" />
<img src="images/output_11_3.png" />
<img src="images/output_11_4.png" />

   The distribution for the target Earning variable is shown below. It is skewed to the left as more students earn less than 50K as we saw in the statistics section above.

<img src="images/output_11_0.png" />


### Algorithms and Techniques

This is a Classification problem as we are trying to predict if sudents will earn at least 50K after 10 years of graduation based on a selected feature set. A decision Tree based model could be used as a starting point and Ensemble methods below can be attempted.

Based on http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html and http://scikit-learn.org/stable/modules/ensemble.html I am thinking of using Ensemble Methods sequentail boosting (AdaBoost, Gradient Boost) and parallel bagging(RandomForest). This is because these classifiers make use of a base classifier (here DecisionTree as default) and improve on that. Error = Bias (where the algorithm cannot learn the target) + Variance (comes from sampling)

RandomForest (Bagging): It is based on fully grown decision trees (low bias, high variance). Bagging reduces error mainly by reducing variance (but not bias) by making the trees uncorrelated. The main weakness is it needs fully grown trees hence increases computational complexity of the model. Can be slow to score as the complexity increases. Its main strength is ability to handle outliers and noise. Also it works fast and off the shelf and typically avoids overfitting. This is a good model for the data as the dataset is not too large or complex and is generally considered a safe bet.

AdaBoost, GradientBoost (Boosting): Boosting is based on weak learners (high bias, low variance).Boosting reduces error mainly by reducing bias (and also to some extent variance), by aggregating the output from many models. Its main weakness is inability to handle outliers and noise and can overfit. It is complex to do tuning with several hyperparameters and finding a stopping point, But it performs well with higher complexities and is fast. They give better results but much harder to train. These are good models for the data as they are very powerful and as the data has been preprocessed for outliers. I guess GradientBoosting is used in most winning Kaggle competitions for a reason.

GradientBoost can be thought of a specific type of AdaptiveBoosting. The main difference is in AdaBoost boosting is done by increasing the weight of incorrect observations so subsequent iterations concentrate on those, whereas in GradientBoost gradient descent logic is used to minimize the loss function when adding trees. In a sense GradientBoost is generic enough to be used for more situations than AdaBoost.

I am thinking of starting with the default parameters for these algorithms, and then fine tune them later to see if they make a difference in accuracy scores.

### Benchmark

Since there are numerous factors that could be used and hence not many standardised studies are available to provide an historic benchmark, I am using Naive predictor methodology where we assume everyone earns above $50,000, as a benchmark model and see
how the above Ensemble models perform.

The scores I got based on his method:
   
   Naive Predictor: [Accuracy score: 0.3348, F-score: 0.3862]
   
   
   
## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

The dataset had 1825 features. Fortunately there was extensive dowcumentation, which I had to read through to shortlist the features needed. 

Next some of the data records did not have information for the target variable, Earnings, as it was privacy protedted. The value was mentioned as "PrivacyProtected". So I had to drop those records.

Some records had several values for the features as invalid (NaN).Had to dop those records to make sure the analysis was close a possible to real data.

The distributions for number of students and university spending on students was highly skewed to the left. Hence had to do log-transformation for these features so they do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers.

<img src="images/output_13_0.png" />

Also applied normalization for the features, since they were numerical, so that each feature is treated equally when applying supervised learners

Finally since I wanted to frame this as a classification problem, encoded the target variable Earnings to 1 if >=50000 and 0 if <50000.

### Implementation

#### Shuffle and Split Data
First step is to split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.

#### Generate a Naive predictor
The purpose of generating a naive predictor is simply to show what a base model without any intelligence would look like. This is becaue I could not find any free historically similar research. In Naive predictor methodology I assume everyone earns above $50,000 and generate accuracy scores.

#### Select Algorithms to test
Using Ensemble Methods sequentail boosting (AdaBoost, Gradient Boost) and parallel bagging(RandomForest) to compare.
To properly evaluate the performance of each model, created a training and predicting pipeline that allows to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. 

- Fit the learner to the sampled training data and record the training time.
- Perform predictions on the test data, and also on the first 300 training points.
- Record the total prediction time.
- Calculate the accuracy score for both the training subset and testing set.
- Calculate the F-score for both the training subset and testing set.

#### Initial model evaluation
Imported the three supervised learning models as discussed. Used the default settings for each model. Calculated the number of records equal to 1%, 10%, and 100% of the training data.

<img src="images/output_30_1.png" />

Best Model : GradientBoost Classifier has the highest score with less time.

GradientBoost has higher F-score and Accuracy than AdaBoostin with Testing set and Trainming set. It also took more time than AdaBoost.
Surprisingly, RandomForest did as good as Boosting alborithms in Training and took less time too. RandomForest did as good as GradientBost with slightly les accuracy scores and slightly more time during testing.

### Refinement

I selected GradientBoost and fine tuned the following parameters:
              learning_rate': [0.1, 0.05, 0.01]
              n_estimators' :[25,50,70]
              max_depth' : [1,3,9]
              
learning_rate : shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.

n_estimators : The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.

max_depth : limits the number of nodes in the tree. 

Unoptimized model
   Accuracy score on testing data: 0.8333
   F-score on testing data: 0.7194

Optimized Model
   Accuracy score on testing data: 0.8444
   F-score on the testing data: 0.7407

Final Optimized Model with Feature Selection: 
Finally wanted to see how the model will perform if only the top five important features were usd to train. 
   Accuracy score on testing data: 0.8556
   F-score on testing data: 0.7634
 
## IV. Results

### Model Evaluation and Validation

The Final optimized model scores are Accuracy 0.8556 and F-score 0.7634
The Optimized model with feature selection is better than the Optimized model which is better than the unoptimized model.
But it has much better scores than benchmark (Accuracy-0.3348, F-score: 0.3862)

#### Results:

|     Metric     | Unoptimized Model | Optimized Model | Optimized(Feature) Model |
| :------------: | :---------------: | :-------------: | :----------------------: |
| Accuracy Score |    0.8333         |     0.8444      |       0.8556             |
| F-score        |    0.7194         |     0.7407      |       0.7634             |


### Justification

Benchmark scores: Accuracy-0.3348, F-score: 0.3862
Final model scores: Accuracy-0.8556, F-score: 0.7634
The final model has an increase of 0.5208 in accuracy score and an increase of 0.3772 in F-score.
Compared to the Benchmark the selected tuned model has gained 155.5% in accuracy score and 97.7% in F-score. Though the benchmark was a Naive model it is still impressive.


## V. Conclusion

### Free-Form Visualization

According to the best fitter model the importance of the features are as below:

<img src="images/output_35_0.png" />

The order of importance of the features make sense. But couple of surprises: 
- College retention rate is ranked the highest as far as getting atleast 50K earning in future. 
- The type of University (public/private) did not seem to matter, so maybe students need not worry about high cost private colleges.

### Reflection

The project can be summarized as following steps:
1. Data Exploration: Cursory investigation of the data to explore the dataset and the featureset and to find degree of correlations between variables.
2. Data Preprocessing: Cleaning the data including formatting and restructuring, normalizing the data and shuffling and splitting the data into training, validation and testing sets.
3. Feature selection: Extract feature importance, select relevant features and create new features if necessary/possible to improve accuracy.
4. Model selection: Experiment with Ensemble algorithms Ada Bost, Random Forest and Gradient Boost to find out the best algorithm for this scenario.
5. Model Tuning: Fine tune the selected model to increase accuracy and performance, including restricting top 5 features.
6. Testing: Test the model on testing dataset and do final model evaluation.

Conceptually the data available for doing this type of investigation is not sufficient as Universities do not want to share this information. Also maybe there are more important factors like college major, but that is out of scope of this investigation. I think within the limitations and constrains of data, this study can help students and parents to think outside the box, to look at other factors when selecting colleges.

### Improvement

To investigate further more algorithms can be tried out like XGBoost. Could have tried fine tuning other parameters too with current model.
Also I had dropped records with features that did not have values. Possibly can try out with filling with average value for the feature, to have a larger dataset. 

