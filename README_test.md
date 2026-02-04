## GEOG0115 Assessment 2: Natural Language Processing for Reported Planning Code Violations
#### Goal: Text analysis of reported Planning Code violations to predict when submissions are for real violations or are false accusations.
#### Notebook 1: Baseline Assessment of Different Models

### Import Libraries, Packages, and Dataset


```python
# Foundational libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Individual scikit-learn tools
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Different machine learning classification models
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
```


```python
# Import prepared dataset
# Anonymized filepath
enforcement = pd.read_csv('/Users/XXX/Documents/XXX/Intro to SDS/SDS Data Report/enforcement_v5.csv')
```

### Exploratory Data Analysis of Categories and Record Statuses


```python
# Check that data uploaded correctly: Columns
enforcement.columns
```




    Index(['RECORD_ID', 'PROJECT_NAME', 'DESCRIPTION', 'DESC-YES', 'RECORD_STATUS',
           'CATEGORY', 'PROJECT_ADDRESS', 'ZIP', 'ZIP_CODE', 'APN', 'BLOCK', 'LOT',
           'YEAR_OP', 'YEAR_CL', 'Closed', 'Date_Op', 'Date_Cl'],
          dtype='object')




```python
# Check that data uploaded correctly: Category Counts
enforcement['CATEGORY'].value_counts()
```




    CATEGORY
    Violation       5567
    No Violation    2884
    Unclear         1405
    In Progress     1096
    Name: count, dtype: int64




```python
# Plot distribution of Categories and Record Statuses
category_counts = (
    enforcement[['CATEGORY', 'RECORD_STATUS']]
    .value_counts()
    .sort_index()
    .reset_index())

category_counts.columns = ['CATEGORY', 'RECORD_STATUS', 'Count']

category_counts.set_index(['CATEGORY', 'RECORD_STATUS']).plot.barh(
    figsize=(8, 8),
    color='black')
```




    <Axes: ylabel='CATEGORY,RECORD_STATUS'>




    
![png](ISDS_Assessment2_Notebook1_files/ISDS_Assessment2_Notebook1_7_1.png)
    



```python
# Plot distribution of Categories totaled up
category_counts = (
    enforcement['CATEGORY']
    .value_counts()
    .sort_index()
    .reset_index())

category_counts.columns = ['CATEGORY', 'Count']

category_counts.set_index('CATEGORY').plot.barh(
    figsize=(8, 8),
    color='black')
```




    <Axes: ylabel='CATEGORY'>




    
![png](ISDS_Assessment2_Notebook1_files/ISDS_Assessment2_Notebook1_8_1.png)
    


### Exploratory Data Analysis of Enforcement Case Descriptions


```python
# Removed ENF cases with no descriptions
enforcement = enforcement[
    enforcement['DESCRIPTION'].notna() &
    (enforcement['DESCRIPTION'].str.strip() != "")]
```


```python
# Check the updated Category Counts
enforcement['CATEGORY'].value_counts()
```




    CATEGORY
    Violation       5537
    No Violation    2808
    Unclear         1396
    In Progress     1093
    Name: count, dtype: int64




```python
# Quantitative analysis of the Descriptions
enforcement['word_count'] = enforcement['DESCRIPTION'].astype(str).str.split().str.len()
enforcement['word_count'].describe()
```




    count    10834.000000
    mean        27.279583
    std         34.743177
    min          1.000000
    25%          7.000000
    50%         16.000000
    75%         35.000000
    max        577.000000
    Name: word_count, dtype: float64




```python
# Prepare the bag-of-words vecotrization and establish a document-term matrix
vectorizer = CountVectorizer(stop_words='english')
desc_words = vectorizer.fit_transform(enforcement['DESCRIPTION'])
```


```python
# Explore the most common words in the Descriptions
word_freq = desc_words.sum(axis=0)

freq_desc_words = pd.DataFrame(
    {'word': vectorizer.get_feature_names_out(),
     'count': word_freq.A1}).sort_values('count', ascending=False)

freq_desc_words.head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16593</th>
      <td>term</td>
      <td>3523</td>
    </tr>
    <tr>
      <th>15717</th>
      <td>short</td>
      <td>3428</td>
    </tr>
    <tr>
      <th>7039</th>
      <td>airbnb</td>
      <td>3098</td>
    </tr>
    <tr>
      <th>11418</th>
      <td>illegal</td>
      <td>3023</td>
    </tr>
    <tr>
      <th>8642</th>
      <td>com</td>
      <td>2552</td>
    </tr>
    <tr>
      <th>17743</th>
      <td>www</td>
      <td>2457</td>
    </tr>
    <tr>
      <th>11353</th>
      <td>https</td>
      <td>2456</td>
    </tr>
    <tr>
      <th>13953</th>
      <td>permit</td>
      <td>2397</td>
    </tr>
    <tr>
      <th>8713</th>
      <td>complaint</td>
      <td>2222</td>
    </tr>
    <tr>
      <th>17088</th>
      <td>unit</td>
      <td>2221</td>
    </tr>
    <tr>
      <th>14889</th>
      <td>rentals</td>
      <td>2150</td>
    </tr>
    <tr>
      <th>17382</th>
      <td>violation</td>
      <td>2089</td>
    </tr>
    <tr>
      <th>16219</th>
      <td>street</td>
      <td>2030</td>
    </tr>
    <tr>
      <th>14419</th>
      <td>property</td>
      <td>1899</td>
    </tr>
    <tr>
      <th>8045</th>
      <td>building</td>
      <td>1868</td>
    </tr>
    <tr>
      <th>15215</th>
      <td>rooms</td>
      <td>1830</td>
    </tr>
    <tr>
      <th>9232</th>
      <td>dbi</td>
      <td>1553</td>
    </tr>
    <tr>
      <th>14888</th>
      <td>rental</td>
      <td>1528</td>
    </tr>
    <tr>
      <th>17211</th>
      <td>use</td>
      <td>1476</td>
    </tr>
    <tr>
      <th>13649</th>
      <td>owner</td>
      <td>1413</td>
    </tr>
    <tr>
      <th>14098</th>
      <td>planning</td>
      <td>1255</td>
    </tr>
    <tr>
      <th>7259</th>
      <td>appears</td>
      <td>1126</td>
    </tr>
    <tr>
      <th>17695</th>
      <td>work</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>11303</th>
      <td>host</td>
      <td>1028</td>
    </tr>
    <tr>
      <th>15774</th>
      <td>sign</td>
      <td>999</td>
    </tr>
    <tr>
      <th>8558</th>
      <td>closed</td>
      <td>956</td>
    </tr>
    <tr>
      <th>6789</th>
      <td>abated</td>
      <td>918</td>
    </tr>
    <tr>
      <th>17780</th>
      <td>yard</td>
      <td>907</td>
    </tr>
    <tr>
      <th>8091</th>
      <td>business</td>
      <td>868</td>
    </tr>
    <tr>
      <th>13800</th>
      <td>parking</td>
      <td>836</td>
    </tr>
  </tbody>
</table>
</div>



### Data Preparation and Cleaning


```python
# Check data type of each columns
enforcement.dtypes
```




    RECORD_ID           object
    PROJECT_NAME        object
    DESCRIPTION         object
    DESC-YES            object
    RECORD_STATUS       object
    CATEGORY            object
    PROJECT_ADDRESS     object
    ZIP                 object
    ZIP_CODE           float64
    APN                 object
    BLOCK               object
    LOT                 object
    YEAR_OP              int64
    YEAR_CL            float64
    Closed              object
    Date_Op             object
    Date_Cl             object
    word_count           int64
    dtype: object




```python
# Convert the Description and Category columns to strings for further analysis
enforcement['DESCRIPTION'] = enforcement['DESCRIPTION'].astype(str)
enforcement['CATEGORY'] = enforcement['CATEGORY'].astype(str)
```


```python
# Keep the records that are Category 'Violation' and 'No Violation' for training and testing
enforcement2 = enforcement[enforcement['CATEGORY'].isin(['Violation', 'No Violation'])]
```


```python
# Confirm that the new split dataset has the correct Category counts
enforcement2['CATEGORY'].value_counts()
```




    CATEGORY
    Violation       5537
    No Violation    2808
    Name: count, dtype: int64




```python
# Review the dataset for formatting and content, confirm readiness for analysis
enforcement2.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RECORD_ID</th>
      <th>PROJECT_NAME</th>
      <th>DESCRIPTION</th>
      <th>DESC-YES</th>
      <th>RECORD_STATUS</th>
      <th>CATEGORY</th>
      <th>PROJECT_ADDRESS</th>
      <th>ZIP</th>
      <th>ZIP_CODE</th>
      <th>APN</th>
      <th>BLOCK</th>
      <th>LOT</th>
      <th>YEAR_OP</th>
      <th>YEAR_CL</th>
      <th>Closed</th>
      <th>Date_Op</th>
      <th>Date_Cl</th>
      <th>word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2074</th>
      <td>2016-000592ENF</td>
      <td>611 42ND AVE</td>
      <td>Illegal Short-Term Vacation Rental</td>
      <td>Yes</td>
      <td>Closed - Abated</td>
      <td>Violation</td>
      <td>611 42ND AVE 94121</td>
      <td>Yes</td>
      <td>94121.0</td>
      <td>1585001F</td>
      <td>1585</td>
      <td>001F</td>
      <td>2016</td>
      <td>2018.0</td>
      <td>Yes</td>
      <td>2016 Jan 13</td>
      <td>2018 Apr 02</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5461</th>
      <td>2018-006350ENF</td>
      <td>1168 FOLSOM ST</td>
      <td>Failure to provide Annual Monitoring Report to...</td>
      <td>Yes</td>
      <td>Closed - Abated</td>
      <td>Violation</td>
      <td>1168 FOLSOM ST 202 94103</td>
      <td>Yes</td>
      <td>94103.0</td>
      <td>3730257</td>
      <td>3730</td>
      <td>257</td>
      <td>2018</td>
      <td>2018.0</td>
      <td>Yes</td>
      <td>2018 Apr 27</td>
      <td>2018 Jul 02</td>
      <td>13</td>
    </tr>
    <tr>
      <th>10259</th>
      <td>2024-005004ENF</td>
      <td>67 Bishop St</td>
      <td>Demolition without permit. Permits 20061103680...</td>
      <td>Yes</td>
      <td>Closed - No Violation</td>
      <td>No Violation</td>
      <td>67 BISHOP ST 94134</td>
      <td>Yes</td>
      <td>94134.0</td>
      <td>6168013</td>
      <td>6168</td>
      <td>013</td>
      <td>2024</td>
      <td>2024.0</td>
      <td>Yes</td>
      <td>2024 May 21</td>
      <td>2024 Jul 05</td>
      <td>12</td>
    </tr>
    <tr>
      <th>413</th>
      <td>13301_ENF</td>
      <td>2 FOLSOM ST</td>
      <td>POPOS Review for compliance</td>
      <td>Yes</td>
      <td>Closed - No Violation</td>
      <td>No Violation</td>
      <td>NaN</td>
      <td>No</td>
      <td>NaN</td>
      <td>3741035</td>
      <td>3741</td>
      <td>035</td>
      <td>2014</td>
      <td>2014.0</td>
      <td>Yes</td>
      <td>2014 Jul 09</td>
      <td>2014 Dec 08</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6365</th>
      <td>2019-005429ENF</td>
      <td>101 San Ramon Way</td>
      <td>Rooms down and work without proper permit appr...</td>
      <td>Yes</td>
      <td>Closed - Abated</td>
      <td>Violation</td>
      <td>101 SAN RAMON WAY 94112</td>
      <td>Yes</td>
      <td>94112.0</td>
      <td>3189002</td>
      <td>3189</td>
      <td>002</td>
      <td>2019</td>
      <td>2019.0</td>
      <td>Yes</td>
      <td>2019 Apr 08</td>
      <td>2019 Jul 15</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



### Prepare Vectorization Model Training with Supervised Text Classification 


```python
# Prepare vectorization parameters, features, and fiiting
tf_vector = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True,
    max_df=0.90,
    min_df=5,
    max_features=500,
    stop_words='english')
```


```python
# Calculate the parameters necessary for the vectorization transformation
enf_TFmatrix = tf_vector.fit_transform(enforcement2['DESCRIPTION'])
```


```python
# Set the focal groups for analysis
X_desc = enf_TFmatrix
Y_cat = enforcement2['CATEGORY']
```


```python
# Split dataset into testing and training sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_desc,
    Y_cat,
    test_size=0.3,
    random_state=80)
```

### Run Different Vectorization Model Trainings with Supervised Text Classification
##### All cells set as Raw, switch to Code one at a time to run each model
##### Results can be compared, cells cannot run simulataneously


```python
# Logistical Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, Y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(max_iter=1000)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LogisticRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(max_iter=1000)</pre></div> </div></div></div></div>


# Multinomial Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train, Y_train)# Support Vector Machine model
clf = LinearSVC()
clf.fit(X_train, Y_train)# Stochastic Gradient Descent Classifier model
clf = SGDClassifier(loss='log_loss', max_iter=1000)
clf.fit(X_train, Y_train)# Complement Naive Bayes model
clf = ComplementNB()
clf.fit(X_train, Y_train)
### Review and Assess Results


```python
# Use the model to predict results
Y_pred = clf.predict(X_test)
```


```python
# Check and print the Accuracy and F1 Scores
acc = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='macro')

print('ACC score is ', acc)
print('F1 score is ', f1)
```

    ACC score is  0.6825079872204473
    F1 score is  0.6101120789694023



```python
# Plot confusion matrix to check accuracy and distribution of results
f, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(confusion_matrix(Y_test, Y_pred),annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")
```




    <Axes: >




    
![png](ISDS_Assessment2_Notebook1_files/ISDS_Assessment2_Notebook1_35_1.png)
    



```python
# Sanity check: run the model on sample text
docs_new = ['there is a confirmed illegal unit. This is reported by the City and is a real violation']
X_new_counts = tf_vector.transform(docs_new)

predictions=clf.predict(X_new_counts)

print(predictions[0])
```

    No Violation

