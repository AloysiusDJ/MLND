# Capstone Project
# Predict College Earning Potential
## Aloysius Joseph

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

Software needed to run and execute [iPython Notebook](http://ipython.org/notebook.html)

Software recommended to install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

### Code

Code is in the `college_data.ipynb` notebook file. Also the included `visuals.py` Python file and the `Most-Recent-Cohorts-All-Data-Elements.csv` dataset file are needed for this project. 

### Run

In a terminal or command window, navigate to the top-level project directory `capstone/` (that contains this README) and run one of the following commands:

```bash
ipython notebook college_data.ipynb
```  
or
```bash
jupyter notebook college_data.ipynb
```

This will open the iPython Notebook software and project file in the browser.

### Data

The dataset consists of approximately XXX data points, with each datapoint having 2060 features. This dataset is provided by the US Department of Education (https://collegescorecard.ed.gov/). But the purpose of this project 9 features have been selected with 1 target variable.

**Features**
- `CONTROL`: Collge type (Public/Private)
- `ADM_RATE`: Admission Rate
- `SATVRMID`: Average SAT score in English
- `SATMTMID`: Average SAT score in Math
- `COSTT4_A`: Average cost to complete education
- `C150_4`: The degree completion rate
- `RET_FT4_POOLED`: Student retention rate
- `NUM4_PUB`: Total number of enrolled students (indicates size)
- `INEXPFTE`: Instructional expenditure per student


**Target Variable**
- `MN_EARN_WNE_INC2_P10`: Mean earnings of students 10 years after entry (<=50K, >50K)
