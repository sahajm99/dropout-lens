# SOURCES.md - what the two UNT course projects asked and claimed

Purpose: a rebuild (dropout-lens) should be able to answer the same questions the two
course projects answered, and check the numbers they reported. Everything below is quoted
or transcribed from the source files; nothing is re-computed. Where a source contradicts
itself or another source, both statements are quoted side by side without adjudication.

Both projects use the same CSV: `dataset.csv` (Kaggle "Higher Education Predictors of
Student Retention", https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention).
The copy in the Spring 2024 folder and the copy in the Fall 2023 folder are byte-identical
(`cmp` reports no difference): 4425 lines including the header (UTF-8 BOM), i.e. 4424 rows,
35 columns. Column list as printed by `student.columns` (Fall 2023 EDA notebook, cell 3):

```
'Marital status', 'Application mode', 'Application order', 'Course',
'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
"Mother's qualification", "Father's qualification", "Mother's occupation",
"Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment',
'International', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
'Unemployment rate', 'Inflation rate', 'GDP', 'Target'
```

Note the dataset's own spelling `Nacionality` (not `Nationality`) - every notebook indexes
the column with that spelling.

---

## Fall 2023: Fundamentals of AI project

Course: CSE-5210, "Team #7". Project title as stated in Checkpoint 1: "Predictive
Analytics for Enhancing Student Success and Retention: Harnessing Learning Data Insights."
Title as stated in `Project description (1).docx`: "Predicting Students' Dropout and
Academic Success".

### Files read (folder `C:/Users/sahaj/OneDrive/Desktop/UNT_Course_Work/UNT Fall 2023/Funda of AI/Project FAI/`)

| File | mtime | What it is |
|---|---|---|
| `Data_visualization_project.ipynb` | 2024-04-22 | 115 cells. EDA + modelling. The complete notebook. |
| `Empirical_Project_EDA.ipynb` | 2024-04-23 | 57 cells. Same EDA code as above, no modelling section. Differences noted below. |
| `Project description (1).docx` | 2023-11-07 | "Project Update Report". Feature list, 53 embedded images (code-value tables, per-feature histograms, EDA charts, model-result screenshots). |
| `Project_proposal_data_visualiztion (1).pdf` | 2023-11-07 | 2-page proposal. Group members listed: Mayur Vora, Kushal Patel, Natarajan Parameswaran, Arnav Sharma. |
| `Checkpoint_3_Draft.docx` | 2023-12-01 | Final-checkpoint draft: Motivation, Problem Definition, Key Issues, Related work, Approach, Conclusion. |
| `conclusion.docx` | 2023-12-01 | Conclusion section only (same text as the Conclusion in Checkpoint_3_Draft). |
| `related work and limitations.docx` | 2023-12-01 | Related-work section only (same text as in Checkpoint_3_Draft). |
| `Script.py` | 2023-11-03 | 20-line skeleton (imports `fetch_20newsgroups`, `CountVectorizer`, `MultinomialNB`; only numbered comments, no code). Unrelated to the dropout dataset. |
| `Done/Project_checkpoint_1_Team_7 (1).pdf` and `Done/Project_checkpoint_1_Team_7_Final Draft.docx` | 2023-09-22 | Checkpoint 1 report. Members (docx): Sahaj Mekala, Mayur Vora, Shiny Shamma Kota, Harshavardhan Aila, Gopinath Reddy, Gangireddygari, Kaushik Apoori, Loka Sai Venkata Tammineedi, Rahul Siddartha Gotti, Mitul Raj Samba. |
| `Done/Project_checkpoint_2_Team_7.pdf` | 2023-11-24 | Checkpoint 2 report (5 pages): Problem Setting, Methodology, Evaluation Plan, limitations paragraph, Preliminary Results with accuracies. |
| `Done/Methodology_Part.docx` | 2023-11-09 | Methodology draft feeding Checkpoint 2. |
| `CSE-5210 Project Checkpoint#{1,2,3} Grade sheet*.docx` | - | Blank rubric templates (no team-specific content). |

The three grade-sheet rubrics contain no project content. `Script.py` contains no
project content.

### Stated research questions / problem statements (verbatim)

Checkpoint 1 (PDF and docx), "2. Problems to be Addressed":

> The primary problems to be addressed in this project are:
> - Predicting student academic success with high accuracy.
> - Identifying students at risk of dropping out early in their academic journey.
> - Developing effective intervention strategies based on predictive insights to improve student retention and performance.

Checkpoint 2 PDF, "Problem Setting":

> The project problem statement based on machine learning algorithm in which the system predicts a student's course dropout or continuation based on provided data.
> Predicting student dropout is classification problem which is closer to an optimization problem rather than a searching problem. The primary goal of this task is to increase the accurate prediction probability of the machine learning models being used.
> The objective of our project is to effectively predict student dropout, the goal is to create machine learning models that can classify students into two groups: "drop out" and "continue" depending on specific characteristics.

Checkpoint_3_Draft.docx, "Problem Definition":

> The problem and objective of this project is to predict whether a student is likely to drop out or graduate based on various features available in the selected dataset. The problem is related to AI as it involves using supervised machine learning algorithms to analyze patterns in dataset and make predictions which are used to take necessary actions by the educational institutions.

Proposal PDF, "Introduction":

> The goal of this project is to explore the complex interplay between student demographics, academic history, financial aid, family income, and other pertinent factors, which contribute to dropout rates and academic achievement. By utilizing machine learning algorithms, we aim to create a robust predictive model that can forecast the likelihood of a student dropping out or succeeding academically.

`Data_visualization_project.ipynb` cell 60 (markdown):

> As we are predicting whether a student will dropout or not so, the number of "Enrolled" student is irrelevant. We only need to know whether a student graduated or dropedout. So, we are dropping the "Enrolled" values and going forward with "Graduate" & "Dropout" values.

Notebook cell 19 (markdown), the EDA plan:

> Since most of the variables in the data set are categorical, we will mainly use bar graphs to visualize them. However, for variables that are discrete or continuous, we will use distribution plots to display their distribution. Additionally, we will utilize correlation heatmaps to examine the relationships between variables in the data.

### Data handling steps in the notebooks (as coded)

Cell numbers refer to `Data_visualization_project.ipynb` unless noted.

| Cell | Code | Printed output |
|---|---|---|
| 0 | `student = pd.read_csv('dataset.csv')` | - |
| 2 | `student.shape` | `(4424, 35)` |
| 6 | `student.info()` | 35 columns, all `4424 non-null`; all `int64` except `Curricular units 1st sem (grade)`, `Curricular units 2nd sem (grade)`, `Unemployment rate`, `Inflation rate`, `GDP` (`float64`) and `Target` (`object`) |
| 7 | `student.isnull().sum()` | `0` for every column |
| 8 | `student.size` | `154840` |
| 9 | `student.describe().T` | see "Numbers claimed" |
| 11 | `student['Target'].value_counts()` | `Graduate 2209`, `Dropout 1421`, `Enrolled 794` |
| 12 | `student['Target'] = LabelEncoder().fit_transform(student['Target'])` | - |
| 13 | `student['Target'].value_counts()` | `2 2209`, `0 1421`, `1 794` |
| 16-17 | `student.drop_duplicates(inplace=True)`; `print(student.duplicated().sum())` | `0` |
| 18 | `student.dropna(inplace=True)` | - |
| 24 | `student.drop(student[student['Target'] == 1].index, inplace = True)` | (drops the label-encoded value 1) |
| 25 | `student['Dropout'] = student['Target'].apply(lambda x: 1 if x==0 else 0)` | - |
| 59 | `student = student.drop(student[student['Target']=='Enrolled'].index)` | - |
| 64 | `x = student.iloc[:, :34].values; x = StandardScaler().fit_transform(x)` | - |
| 65 | `y = student['Target'].values` | `array(['Dropout', 'Graduate', 'Dropout', ..., 'Dropout', 'Graduate', 'Graduate'], dtype=object)` |
| 67 | `encoder = LabelEncoder(); student['Target'] = encoder.fit_transform(student['Target'])` | - |
| 69 | `X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)` | (this split is overwritten by cell 76) |
| 72 | `X = student.drop(columns=['Nacionality','International','Target'], axis=1); Y = student['Target']` | - |
| 74 | `print(Y, Y.shape)` | `Name: Target, Length: 3630, dtype: int64 (3630,)`; first values `0 0, 1 1, 2 0, 3 1, 4 1` |
| 76 | `X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)` | - |
| 77 | `print(X.shape, X_train.shape, X_test.shape)` | `(3630, 33) (2904, 33) (726, 33)` |

Observation on execution order: cell 12 label-encodes `Target` to 0/1/2, yet cell 65
later prints `Target` as the strings `'Dropout'`/`'Graduate'`, and cell 59 filters on the
string `'Enrolled'`. The stored outputs therefore do not correspond to a single top-to-bottom run.

Differences in `Empirical_Project_EDA.ipynb`: same cells 0-21; gender pie (cell 22) uses
`labels = ['0', '1']` instead of `['Female', 'Male']`; no `hue_order` on the countplots;
`Dropout` column derived (cell 24) without first dropping any Target value; ends at the
heatmap (cell 55) with the same conclusion; no modelling cells.

### Visualisations made, chart by chart

All from `Data_visualization_project.ipynb` (same charts appear in `Empirical_Project_EDA.ipynb`).
"Claim" is the markdown text that follows the chart, verbatim.

| # | Cell | Chart | x | y | Claim as written |
|---|---|---|---|---|---|
| 1 | 10 | `student.plot()` line plot of every column | row index | column values | (none) |
| 2 | 14 | `sns.distplot(student['Target'], color="red")` | Target (label-encoded 0/1/2) | Density | (none) |
| 3 | 15 | `sns.countplot(data=student, x="Target")` titled 'Target' | Target | count | (none) |
| 4 | 20 | `plt.pie(student_target, labels=['Graduate','Dropout','Enrolled'], explode=(0.1,0.1,0.0), autopct='%1.2f%%')` titled 'Percentage of Student Target (Education Status)' | - | share of rows | "Approximately 50% of students in the data have graduated." Rendered slices: Graduate 49.93%, Dropout 32.12%, Enrolled 17.95%. |
| 5 | 23 | `plt.pie(student['Gender'].value_counts(), labels=['Female','Male'], explode=(0.1,0.0), autopct='%1.2f%%')` titled "Gender" | - | share | (none). Rendered (EDA-notebook version with labels '0','1'): 0 = 64.83%, 1 = 35.17%. |
| 6 | 26 | 35 `sns.distplot(student.iloc[:, i], color='green')` panels in a 12x3 grid | each column | density | (none) |
| 7 | 27 | `sns.countplot(x='Gender', hue='Target', hue_order=['Dropout','Enrolled','Graduate'])`, xticks `['Female','Male']` | Gender | Number of Students | "According to the data, a higher number of graduates are female. However, females also have the highest number of dropouts, although the difference compared to males is small." |
| 8 | 29 | `sns.countplot(x='Marital status', hue='Target', ...)`, xticks `['Single','Married','Widower','Divorced','Facto Union','Legally Seperated']` | Marital Status | Number of Students | "Regarding marital status, the majority of both graduates and dropouts are single." |
| 9 | 31 | stacked `barh` of `groupby(['Course','Target']).size()` pivot, index renamed to course names, sorted by total | Number of Students | Course | "Nursing course produced the highest number of graduates while management course has the highest number of droputs." |
| 10 | 33 | stacked `barh` for Nationality = Portuguese only, title 'Data for Portuguese' | Number of Students | Nationality | - |
| 11 | 34 | stacked `barh` for all other nationalities, title 'Data for All Other Index Values' | Number of Students | Nationality | "The plot shows that the majority of the students in the dataset are Portuguese, which accounts for the highest frequency among all the nationalities." |
| 12 | 36 | `sns.countplot(x='Displaced', hue='Target', ...)`, xticks `['No','Yes']` | Displaced | Number of Students | "Students who already graduated are mostly displaced students." |
| 13 | 38 | `sns.countplot(x='International', hue='Target', ...)`, xticks `['No','Yes']` | International | Number of Students | "Since Portuguese students dominate the data, it is reflected to numbers of students vs. international bar plot." |
| 14 | 40 | stacked `barh` for Previous qualification = Secondary Education only, title 'Data Secondary Education' | Number of Students | Previous Qualification | - |
| 15 | 41 | stacked `barh` for all other previous qualifications | Number of Students | Previous Qualification | "Most of the students in the data finished secondary education." |
| 16 | 43 | `sns.displot(x='Age at enrollment', kde=True)` | Age at Enrolment | Number of Students | "The distribution of age at enrolment is positively skewed, indicating that the majority of students enrolled at a relatively young age. The mean age at enrolment is approximately 23 years old, with the most frequent age range falling between 19 to 25 years old." |
| 17 | 45 | stacked `barh`, Father's occupation, renamed, top rows only (`sorted[36:]`) | Number of Students | Father's Occupation | - |
| 18 | 46 | stacked `barh`, Mother's occupation, renamed, top rows only (`sorted[22:]`) | Number of Students | Mother's Occupation | "Highest number of students who graduated and dropped out have parents who are unskilled workers." |
| 19 | 48 | `sns.countplot(x='Educational special needs', hue='Target', ...)`, xticks `['No','Yes']` | Educational Special Needs | Number of Students | - |
| 20 | 49 | `sns.countplot(x="Debtor", hue='Target', ...)`, xticks `['No','Yes']` | Debtor | Number of Students | - |
| 21 | 50 | `sns.countplot(x="Tuition fees up to date", hue='Target', ...)`, xticks `['No','Yes']` | Tuition Fees Up to Date | Number of Students | - |
| 22 | 51 | `sns.countplot(x="Scholarship holder", hue='Target', ...)`, xticks `['No','Yes']` | Scholarship Holder | Number of Students | "In terms of other socioeconomic status, most students who graduated and dropped do not have special needs. Also, they are non-debtors and their tuition fees are up to date. Yet, these students are non-scholarship holders." |
| 23 | 53 | `sns.displot(x="Unemployment rate", kde=True)` | Unemployment Rate | Number of Students | "The majority of the data points in the unemployment rate distribution fall within the range of 9 to 13." |
| 24 | 55 | `sns.heatmap(student.corr(), cmap='coolwarm')` titled 'Correlation Heatmap between Variables' | all columns | all columns | "Apparently, correlation between features are low except for Nationality and International. Hence, we can dropped these features in the regression model for predicting student target." |
| 25 | 82, 86, 90, 94, 98, 101 | `ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Dropout', 'Dropout'])` per model | predicted | true | - |
| 26 | 103 | `PrecisionRecallDisplay.from_predictions(Y_test, y_pred_*)` for GNB, LR, RF, XGB, SVC, MLP on one axis, title "Precision-Recall Curve" | Recall | Precision | (none) |
| 27 | 105 | `RocCurveDisplay.from_predictions(Y_test, y_pred_*)` same six models, title "ROC Curve" | FPR | TPR | (none) |

Charts 26-27 are built from the hard class predictions (`y_pred_*`), not from
probabilities.

`Project description (1).docx` additionally embeds one histogram per feature (34 images)
under "Task 3: For each feature you need to provide exploratory statistics (mean, std,
histogram)". Each image is captioned with the feature's mean and std, e.g.
"Marital status: mean=1.1785714285714286, std=0.60574694613071" and
"Age at enrollment: mean=23.265144665461122, std=7.587815615029815". The values match
`student.describe()` (table in "Numbers claimed"). The docx also embeds the charts 4, 5, 7,
8, 9, 11, 12 and 24 above under "Exploratory Data Analysis Results" and "Heatmap to find
Correlation to the Target Variable".

### Models trained and metrics reported

Setup (cells 72-78): features = all columns except `Nacionality`, `International`,
`Target` (33 features); rows = Dropout + Graduate only (3630); `train_test_split(X, Y,
test_size=0.2, random_state=3)` -> train 2904, test 726. The `perform()` helper prints
`precision_score`, `recall_score`, `f1_score` with `average='micro'` plus `accuracy_score`,
the confusion matrix, and `classification_report`. Because `average='micro'` is used, the
four headline numbers are identical for each model.

Metrics exactly as printed (test set, 726 rows; class 0 support 290, class 1 support 436):

| Model (constructor as coded) | Precision/Recall/Accuracy/F1 (micro) | Confusion matrix | classification_report rows (precision, recall, f1) |
|---|---|---|---|
| `GaussianNB()` | `0.8388429752066116` | `[[219  71] [ 46 390]]` | 0: 0.83 0.76 0.79; 1: 0.85 0.89 0.87; accuracy 0.84; macro avg 0.84 0.82 0.83; weighted avg 0.84 0.84 0.84 |
| `LogisticRegression()` (emitted `ConvergenceWarning: lbfgs failed to converge (status=1)`) | `0.9035812672176309` | `[[243  47] [ 23 413]]` | 0: 0.91 0.84 0.87; 1: 0.90 0.95 0.92; accuracy 0.90; macro 0.91 0.89 0.90; weighted 0.90 0.90 0.90 |
| `RandomForestClassifier()` | `0.9146005509641874` | `[[246  44] [ 18 418]]` | 0: 0.93 0.85 0.89; 1: 0.90 0.96 0.93; accuracy 0.91; macro 0.92 0.90 0.91; weighted 0.92 0.91 0.91 |
| `XGBClassifier()` (n_estimators=100) | `0.8980716253443526` | `[[241  49] [ 25 411]]` | 0: 0.91 0.83 0.87; 1: 0.89 0.94 0.92; accuracy 0.90; macro 0.90 0.89 0.89; weighted 0.90 0.90 0.90 |
| `SVC()` | `0.8815426997245179` | `[[219  71] [ 15 421]]` | 0: 0.94 0.76 0.84; 1: 0.86 0.97 0.91; accuracy 0.88; macro 0.90 0.86 0.87; weighted 0.89 0.88 0.88 |
| `MLPClassifier()` (emitted `ConvergenceWarning: ... Maximum iterations (200) reached`) | `0.8966942148760331` | `[[241  49] [ 26 410]]` | 0: 0.90 0.83 0.87; 1: 0.89 0.94 0.92; accuracy 0.90; macro 0.90 0.89 0.89; weighted 0.90 0.90 0.90 |
| `xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000)` ("XGB Logistic Regression", cells 107-111) | `Accuracy: 0.9008264462809917` | (not printed) | (not printed) |

Which class is 0 and which is 1 in the confusion matrices: the notebook says (cell 58
markdown) "The labels dropout and graduate become 0 and 1, respectively." The
`ConfusionMatrixDisplay` in the same notebook is labelled `display_labels=['Non-Dropout',
'Dropout']`, i.e. it names index 0 "Non-Dropout". Both statements are in the source; they
assign opposite meanings to index 0. No random seed is set for RF, XGB, SVC or MLP.

"Creating a System for Prediction" (cells 113-114):

```
# The input data is the 192nd record in the student_data dataset disregarding the Nationality and International record
input_data = (1, 1, 2, 14, 1, 1, 1, 3, 5, 4, 0, 0, 0, 1, 0, 0, 19, 0, 5, 5, 5, 13, 0, 0, 5, 5, 5, 13.2, 0, 9.4, -0.8, -3.1,0)
...
prediction = bin_log.predict(input_data_reshaped)
```
Output: `[1]` / `The initial value is  1`; reference `print(student['Target'].iloc[195])` -> `1`.

Earlier/other reported model numbers (screenshots and reports, not in the notebook):

- `Project description (1).docx`, image after "XGB Logistic Regression": the same
  `xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000)` cell, whose fitted
  repr shows `objective='multi:softprob'`, followed by `Accuracy: 0.7898305084745763`.
- `Project description (1).docx`, image after "Then we trained various models to choose
  the best model for our target variable and we got random forest as the best model with
  following results:":
  ```
  rf_best = RandomForestClassifier(n_estimators=150, max_depth=4)
  ...
  print('Precision:', precision_score(Y_test,lr_preds, average=None))
  print('Recall:', recall_score(Y_test, lr_preds, average=None))
  print('F1-score:', f1_score(Y_test, lr_preds, average=None))
  print("Accuracy:", accuracy)
  Precision: [0.79183673 0.42156863 0.79553903]
  Recall: [0.74615385 0.2654321  0.92440605]
  F1-score: [0.76831683 0.32575758 0.85514486]
  Accuracy: 0.727683615819209
  ```
  (three-class arrays; precision/recall/F1 are computed on `lr_preds`, accuracy on `y_pred`.)
- Checkpoint 2 PDF, "Preliminary Results": "Logical regression(91.75%), Random
  Forest(89.97%), XGBoost Classifier(90.31%), support vector classifier(91.32%) are the
  accuracy results. A 100% though sounds good is not desirable value as it implies
  overfitting. The models have an average 90% accuracy". Same PDF: "HyperParameters
  considered as of now: train to test split ratio is 3:1, random state = 3".
- Checkpoint 1 draft docx, related work: Kiss et al. "achieving a prediction accuracy of
  85.3%"; Magalhaes et al. "achieved an average accuracy of 75.4%".

Models listed in the reports but with no code in the notebooks: "decision trees"
(Checkpoint 2 evaluation plan, Checkpoint_3_Draft), "linear regression"
(Checkpoint_3_Draft conclusion), hyperparameter tuning via "grid search and random search"
(Checkpoint_3_Draft), K-fold cross-validation (Checkpoint 2).

### Stated limitations (verbatim)

Checkpoint 2 PDF, page 4:

> For this Project, we have a big dataset that has 40-odd features. We are making decisions based on many target variables so if any of those variables have null values then the predictions will be biased to some other feature variables. We cannot be sure that the provided data does not have any errors or inconsistencies because that can lead us to inaccurate training, and we may face the problems of overfitting or underfitting. The given scenario can also be obtained by an imbalanced dataset too especially when we just focus on the accuracy we want to achieve. Deciding relevant features and transforming them according to the need is also an important aspect, Inaccurate selection can lead to suboptimal models. We may face the problem of overfitting if complex models fail to generalize to new data. Underfitting can also take place if we use very simple models which may not capture important patterns for predicting the results. Complex models such as Deep learning can easily lose interpretability and that makes it difficult to understand the meaning or outcomes behind any predictions.
> We believe that the effectiveness or accuracy of the model may differ in real-world scenarios from the evaluation we performed for this project.

Checkpoint_3_Draft.docx, "Key Issues":

> Key Challenges we faced in this project would be identifying relevant features by performing Exploratory Data Analysis, handling missing data and encoding text data. Model Training and Hyperparameter tuning was relatively simple as the pre defined functions in Sci-Kit Learn were used to implement them.

Checkpoint_3_Draft.docx / conclusion.docx:

> While these initial steps gave us many valuable insights, there is room for many improvements and refinement in future versions of this project.
> ... we can achieve more refined and precise predictions from some advanced feature engineering techniques, exploration of diverse or hybrid machine learning methods, and an adequate hyperparameter tuning process. Furthermore, we think that the integration of ensemble methods and the mitigation of biases will help us to develop a more reliable and equitable model.
> The project's future trajectory will depend critically on addressing ethical issues, guaranteeing model interpretability, and conducting ongoing monitoring and updates.

`related work and limitations.docx` (limitations attributed to prior work, and the claimed response):

> Aulck, Lovenoor S. et al. "Predicting Student Dropout in Higher Education." ArXiv abs/1606.06364 (2016) - We noticed that this work is mainly focused on using demographic and academic performance features for prediction. It has not explored the complex and important behavioral and socioeconomic variables that can affect students' outcomes. In our approach, we have gone through a comprehensive feature engineering process ... We also considered socio-economic factors and behavioral patterns along with external factors such as family income, family status, and so on
> J. Xu, K. H. Moon, and M. van der Schaar, "A Machine Learning Approach for Tracking and Predicting Student Performance in Degree Programs," IEEE JSTSP 11(5), 2017 - Class inequalities are widespread in educational databases, and this study encountered difficulties in resolving them. ... Our approach includes a step to evaluate and potentially address class imbalances. A more robust model can be produced by using strategies like undersampling, oversampling, or changing class weights during model training

Checkpoint 1 per-model limitations: Logistic Regression - "complicated non-linear
relationships in the data may not be adequately captured"; Random Forest - "If not
properly tuned, Random Forest can overfit. It can also be computationally demanding";
Gradient Boosting - "Their training might be computationally expensive and they need
careful hyperparameter tuning. If overfitting isn't controlled, it's a problem."; SVM -
"SVM performance could decrease with large datasets and can be sensitive to the kernel
selection. Without a suitable approach, they might not be the best option for unbalanced
datasets."

Pre-processing claims: Checkpoint 1 - "The selected dataset does not require any
pre-processing as the data for columns like marital status, parent's occupation etc., has
already been encoded"; Checkpoint 2 - "The dataset used here doesn't have any missing
values and all the features are encoded, scaled and numeric. The target variable needs to
be encoded as a numeric value which is the only preprocessing step necessary for this
dataset."

"Things to work on Future" (`Project description (1).docx`): "Exploratory data analysis;
Use multiple machine learning algorithm.; Hyperparameter tuning; Run machine learning algo
to if model is overfitting.; Compare accuracy of our model."

---

## Spring 2024: Empirical Analysis project

Title: "Higher Education Student Retention Predictors", "GROUP-4". Members (all pptx):
Gowthami Kasi, Sahaj Mekala, Harshavardhan Aila, Saieesh Kumar Bhandar. Roles (first
pptx, slide 2): Gowthami Kasi - Report Writing; Sahaj Mekala - Statistical Tests and
Interpretation; Harshavardhan Aila - Data Analysis; Saieesh Kumar Bhandar - Data
Visualization and future scope. GitHub repo named on slide 3 of the first pptx:
https://github.com/HarshavardhanAila/Emperical-Analysis-Project_Student-retention-

### Files read (folder `C:/Users/sahaj/OneDrive/Desktop/UNT_Course_Work/UNT Spring 2024/Empirical Analysis/`)

| File | mtime | What it is |
|---|---|---|
| `Project/Empirical_Analysis_Project_Codebook.ipynb` | 2024-04-24 | 25 cells: target encoding, 8 chi-square tests, 1 t-test, 1 ANOVA, a "Linear Regression" heading with an empty cell. |
| `Project/Empirical_Saieesh.ipynb` | 2024-04-23 | 22 cells: superset of the Codebook - 16 chi-square, 16 t-tests, 9 ANOVAs, with conclusion comments. |
| `Project/Empirical increment EDA.docx` | 2024-03-25 | 10 headings, each followed by one chart image (no prose). |
| `Project/Emperical Analysis_Student Redention.pptx` | 2024-03-24 | 13 slides: proposal deck. |
| `Project/Emperical Analysis_Student Redention_Project Update.pptx` | 2024-03-27 | 18 slides: same text plus 8 "Analysis and Visualization" image slides. |
| `Higher Education Student Retention Predictors_Final Slides.pptx` | 2024-04-21 | 12 slides. Only titles are present; every content placeholder is empty (verified in the slide XML). Slide 8 holds a 4x3 table whose header row reads `Chi-square | T-test | anova` and whose three body rows are blank. |
| `Project/dataset.csv` | 2024-02-25 | identical to the Fall 2023 copy. |
| `Project/video1969337353.mp4` | 2024-03-28 | not read. |
| `Project/.ipynb_checkpoints/Empirical_Analysis_Project_Codebook-checkpoint.ipynb` | - | 72 bytes, empty. |

### Target encoding used by both Spring notebooks (cell 0)

```
df['Target'] = df['Target'].map({'Graduate': 0, 'Dropout': 1, 'Enrolled': 0})
```
So `Target == 1` is Dropout and `Target == 0` is Graduate + Enrolled. Every test below
uses all 4424 rows (no rows are dropped). The slides describe the target differently:
"Target Variable: Student Retention (0 = dropped out, 1 = retained)" (first and update
pptx, "Dataset Description" slide).

### Hypothesis tests run

All statistics below are the printed cell outputs. In `Empirical_Saieesh.ipynb` cell 18
the same numbers are also pasted back into the code as comments.

Decision rule as stated (Codebook cell 20, markdown): "Based on the p-value in each case,
if p < 0.05, we reject the null hypothesis, indicating that the variables are significantly
associated. Otherwise, if p >= 0.05, we fail to reject the null hypothesis, suggesting no
significant association between the variables."

#### Chi-square tests of independence - `chi2_contingency(pd.crosstab(df[col], df['Target']))`

H0/H1 as written for each: "`<variable>` and dropout/graduation rates are independent" /
"... are dependent". "Conclusion comment" is the comment written under the test in
`Empirical_Saieesh.ipynb` cell 18, verbatim; the eight marked "Codebook" also appear in
`Empirical_Analysis_Project_Codebook.ipynb`.

| # | Variable (column) | Chi-square test statistic | P-value | Conclusion comment as written (Saieesh cell 18) | Also in Codebook |
|---|---|---|---|---|---|
| 1 | Application mode | `399.11637213617763` | `2.8369904449819405e-74` | "if p >= 0.05, we fail to reject the null hypothesis; indicating that Application mode and dropout/graduation rates are not associated with each other." | no |
| 2 | Daytime/evening attendance | `28.11761592819376` | `1.1416199101077951e-07` | "... attendance schedule and dropout/graduation rates are not associated with each other." | no |
| 3 | Age at enrollment | `475.71777022704435` | `2.843967994411833e-73` | "... age at enrollment and dropout/graduation rates are not associated with each other." | no |
| 4 | International | `0.3430007141187338` | `0.5581022527918915` | "... international status and dropout/graduation rates are not associated with each other." | no |
| 5 | Curricular units 1st sem (grade) | `1558.3158572814536` | `1.1508663025593669e-50` | "... grades in first semester and dropout/graduation rates are not associated with each other." | no |
| 6 | Curricular units 2nd sem (grade) | `1883.8145942885235` | `5.862394128264281e-92` | "... grades in second semester and dropout/graduation rates are not associated with each other." | no |
| 7 | Unemployment rate | `32.9533423734193` | `0.00013607310561987608` | "if p < 0.05, we reject the null hypothesis; indicating that unemployment rate and dropout/graduation rates are significantly associated with each other." | no |
| 8 | Inflation rate | `28.161356476872218` | `0.00044452044442922135` | "if p < 0.05, we reject the null hypothesis; indicating that Inflation rate and dropout/graduation rates are significantly associated with each other." | no |
| 9 | Marital status | `59.44536549353541` | `1.582435922288625e-11` | "... marital status and dropout/graduation rates are not associated with each other." Codebook cell 4 markdown instead: "Based on the p-value, if p < 0.05, we reject the null hypothesis, indicating that marital status and dropout/graduation rates are significantly associated." | yes (cell 3) |
| 10 | Gender | `183.16407489465018` | `9.87658631679028e-42` | "... gender and dropout/graduation rates are not associated with each other." | yes (cell 5) |
| 11 | Scholarship holder | `265.10369150465624` | `1.324567089306277e-59` | "... scholarship holder status and dropout/graduation rates are not associated with each other." | yes (cell 7) |
| 12 | Nacionality | `17.666980618595606` | `0.6093354711054288` | "... nationality and dropout/graduation rates are not associated with each other." | yes (cell 9) |
| 13 | Previous qualification | `202.61780570673022` | `2.3440454961507005e-34` | "... previous qualification and dropout/graduation rates are not associated with each other." | yes (cell 11) |
| 14 | Displaced | `50.410213409673304` | `1.2474332333038966e-12` | "... displaced status and dropout/graduation rates are not associated with each other." | yes (cell 15) |
| 15 | Debtor | `231.2799041609192` | `3.1348742631274317e-52` | "... debtor status and dropout/graduation rates are not associated with each other." | yes (cell 17) |
| 16 | Educational special needs | `0.0012812523104101163` | `0.9714461509005853` | "... Educational special needs and dropout/graduation rates are not associated with each other." | yes (cell 19) |

Note on the comments: rows 1-6 and 9-16 all carry the same template sentence ("if p >=
0.05, we fail to reject ... not associated") regardless of the printed p-value; only rows
7 and 8 carry the "p < 0.05 ... significantly associated" sentence. The Codebook's own
generic rule (quoted above) is the only place the p < 0.05 branch is applied in general.
Tests 3, 5, 6, 7 and 8 run `pd.crosstab` on numeric columns (age, grades, rates), so each
distinct value becomes a contingency row.

#### Two-sample t-tests - `ttest_ind(df[df['Target']==1][col], df[df['Target']==0][col])`

(`Empirical_Saieesh.ipynb` cell 19; test 1 also in Codebook cell 22 and Saieesh cell 16.)
H0/H1 as written: for test 1, "Mean age at enrollment is the same for dropout and graduate
students." / "... is different between ..."; for all others the same
"independent"/"dependent" wording as the chi-square tests.

| # | Column | T-test statistic | P-value | Conclusion comment as written |
|---|---|---|---|---|
| 1 | Age at enrollment | `17.479079612784894` | `3.3227218222273917e-66` | "if p >= 0.05, we fail to reject the null hypothesis; indicating that Age at enrollment and dropout/graduation rates are not associated with each other." (Codebook cell 22: "Based on the p-value, if p >= 0.05, we fail to reject the null hypothesis.") |
| 2 | Daytime/evening attendance | `-5.370445652546331` | `8.25718601372232e-08` | "... attendance schedule and dropout/graduation rates are not associated with each other." |
| 3 | Application mode | `12.792352219338833` | `8.073698868233097e-37` | "... Application mode and dropout/graduation rates are not associated with each other." |
| 4 | International | `-0.6889345481917433` | `0.49090060139273606` | "... international status and dropout/graduation rates are not associated with each other." |
| 5 | Curricular units 1st sem (grade) | `-36.450580077849715` | `1.3387168593456852e-254` | "... grades in first semester and dropout/graduation rates are not associated with each other." |
| 6 | Curricular units 2nd sem (grade) | `-46.34711948995211` | `0.0` | "if p < 0.05, we reject the null hypothesis; indicating that grades in second semester and dropout/graduation rates are significantly associated with each other." |
| 7 | Unemployment rate | `0.8631909185930985` | `0.38807931933391604` | "... Unemployment rate and dropout/graduation rates are not associated with each other." |
| 8 | Inflation rate | `1.8510949132028756` | `0.06422253536081886` | "... Inflation rate and dropout/graduation rates are not associated with each other." |
| 9 | Marital status | `6.259228053537021` | `4.235899474916463e-10` | "... marital status and dropout/graduation rates are not associated with each other." |
| 10 | Gender | `13.855784810263657` | `9.085401091612991e-43` | "... Gender and dropout/graduation rates are not associated with each other." |
| 11 | Scholarship holder | `-16.83000741237761` | `1.1766316558275064e-61` | "... Scholarship holder status and dropout/graduation rates are not associated with each other." |
| 12 | Nacionality | `-0.10447803164821953` | `0.9167947393200735` | "... Nationality and dropout/graduation rates are not associated with each other." |
| 13 | Previous qualification | `6.116282551659485` | `1.0405599865233786e-09` | "... Previous qualification and dropout/graduation rates are not associated with each other." |
| 14 | Displaced | `-7.172107179368192` | `8.616292530904895e-13` | "... Displacement status and dropout/graduation rates are not associated with each other." |
| 15 | Debtor | `15.673149877954302` | `6.367415905172124e-54` | "... Debtor status and dropout/graduation rates are not associated with each other." |
| 16 | Educational special needs | `0.18656765119943888` | `0.8520081902400654` | "... educational needs and dropout/graduation rates are independent and not asssociated." |

#### One-way ANOVA - `f_oneway(*[df[df[group]==g][measure] for g in df[group].unique()])`

(`Empirical_Saieesh.ipynb` cell 20; test 1 also in Codebook cell 22 and Saieesh cell 16.)
H0/H1 are quoted because the wording flips between tests.

| # | Grouping column | Measure | ANOVA F-statistic | P-value | H0 as written | Conclusion comment as written |
|---|---|---|---|---|---|---|
| 1 | Nacionality | Curricular units 1st sem (grade) | `0.4843581948628184` | `0.9733506184234109` | "Mean grades in the first semester are the same across different nationalities." | "if p >= 0.05, we fail to reject the null hypothesis; indicating that grades in first semester are different across different nationalities." (Codebook: "Based on the p-value, if p >= 0.05, we fail to reject the null hypothesis.") |
| 2 | Course | Curricular units 1st sem (grade) | `72.50240668106373` | `2.1296260106182052e-209` | "Mean grades in the first semester are different across different courses." | "if p >= 0.05, we fail to reject the null hypothesis; indicating that grades in first semester are different across different courses." |
| 3 | Mother's qualification | Curricular units 2nd sem (grade) | `4.466196992890782` | `5.460451051238551e-14` | "Mean grades in the second semester are different across different mother's qualifications." | "... indicating that mean grades in second semester are different across mothers qualifications." |
| 4 | Gender (`df['Gender'] == 'Male'` / `== 'Female'`) | Age at enrollment | `nan` | `nan` | "Mean age at enrollment is different between genders." | comment unfinished: "if p < 0.05, we reject the null hypothesis ; indication that". Kernel emitted `DegenerateDataWarning: at least one input has length 0` (Gender is an int column, so the string comparisons select no rows). |
| 5 | Previous qualification | Curricular units 2nd sem (grade) | `10.545973117599466` | `5.659453546325767e-27` | "Mean grades in the second semester are different across different previous qualifications." | "... indicating that grades in second semeester are significantly different across previous qualifications." |
| 6 | Father's occupation | Curricular units 1st sem (grade) | `1.8852291581832619` | `0.00033657613796549374` | "Mean grades in the first semester are different across different father's occupations." | "if p < 0.05, we reject the null hypothesis ; indication that grades in first semester is same across different fathers occupations." |
| 7 | Nacionality | Age at enrollment | `0.959284712139499` | `0.5099535034743299` | "Mean age at enrollment is different across different nationalities." | "if p >= 0.05, we fail to reject the null hypothesis; indicating that mean age at enrollment is different across nationalities." |
| 8 | Marital status | Curricular units 2nd sem (grade) | `5.570877039879294` | `4.0251711413251806e-05` | "Mean grades in the second semester are different across different marital statuses." | "... indicating that grades in second semester are significantly different across different marital status." |
| 9 | Debtor (`==1` vs `==0`) | Curricular units 1st sem (grade) | `48.511294100941576` | `3.767937195653264e-12` | "Mean grades in the first semester are different between debtors and non-debtors." | "... indicating that Grades in first semester are significantly different betweeen debtors and non debtors." |

Not run: the Codebook has a markdown heading "Linear Regression" (cell 23) followed by an
empty code cell; no regression or correlation-coefficient code exists in either Spring
notebook, although the slides list "Correlation Analysis" as performed (see below).

### The slides' claims (verbatim; numbers in bold)

First pptx (2024-03-24) and Project Update pptx (2024-03-27), "Dataset Description" slide:

> Dataset: Higher Education Student Retention Predictors
> Source: Kaggle (https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data)
> Variables: **35** (including target variable)
> Samples: **4424**
> Target Variable: Student Retention (**0 = dropped out, 1 = retained**)

(The two pictures on that slide are the `describe()` table split in two halves; values
identical to the table in "Numbers claimed".)

Abstract (both decks):

> The goal of this project is to estimate Student's retention in universities using a student dropout dataset obtained from Kaggle.
> Here we are trying to perform an Exploratory data analysis(EDA) and apply statistical tests which helps us identify the relationships and patterns across the data which will eventually enhance the performance and improve the predictions which are accurate.
> By analyzing different factors which contribute to student retention, we aim to provide the valuable insights for Universities .

Executive Summary (first deck only):

> In order to find significant patterns and relationships in the data that will give us relevant information for making well-informed judgements, we plan to apply statistical tests and exploratory data analysis (EDA).
> In order to establish an environment that encourages student success and reduces dropout rates, we look at the factors that lead to student dropout in an effort to provide institutions with targeted solutions.

Project workflow summary (both decks): "Dataset: collected data from kaggle; Preprocessing
of data: to handle the missing values and errors in dataset, additionally we can remove
duplicate data.; Exploratory data analysis: we can find relationship between variables,
furthermore we can get visualize the data and find patterns.; Selection of features: we
can use technical correlation analysis.; Selection of training dataset and test dataset by
splitting.; Cross validation techniques are useful for model evaluation.; Deployment and
improve the accurate performance of analysis."

Statistical tests (first deck, "Statistical Tests:"; update deck, "Statistical Tests performed:" lists only item 1):

> 1. Correlation Analysis: To identify relationships between variables and discover for any significant patterns or trends.
> 2. Chi-Square Test: To compare categorical variable and assess for any difference in trends.
> 3. ANOVA: Using Analysis of Variance we try to compare the means of categorical variables.

Milestones (both decks): "Week 1: Data preprocessing(cleaning) and Exploratory Data
Analysis(EDA). Week 2: Correlation Analysis and Regression Analysis. Week 3: ANOVA. Week
4: Interpret Results and Visualization. Week 5: Report Writing and Final Presentation."

Tools (first deck): "Google Colab; Kaggle for dataset; Github for collaboration; Python
Libraries: Numpy, Pandas, Scipy, Sklearn, Matplotlib and Seaborn (for visualization)".

Project Future Scope (first deck):

> We plan to enhance this project by using several machine learning models, incorporating predictive modeling and perform real time analysis while integrating with other data sources and making it accessible to a broader audience using interactive visualization and do the long term trend monitoring and analysis to handle sensitive data along with optimizing models performance by providing feedbacks and using Chi-Squared Test to compare several models for performance and ensure that it aligns with our future scope

"Analysis and Visualization" image slides (Update deck slides 7-14; first deck slide 8):
target pie (Graduate **49.93%**, Dropout **32.12%**, Enrolled **17.95%**); gender pie
(0 = **64.83%**, 1 = **35.17%**); Target by Gender countplot; Marital Status countplot;
Course stacked barh; correlation heatmap (default colormap, drawn in Colab); a seaborn
hex `jointplot` of Unemployment rate (x) vs Inflation rate (y) (shown twice); nationality
"Data for All Other Index Values" barh; Previous qualification "Data for All Other Index
Values" barh; Age at enrollment histogram+KDE. No numeric claims are printed on these slides.

Final Slides deck (2024-04-21): slide titles only - "Abstract:", "Dataset
specification/description:", "Exploratory Data Analysis:", "Potential Statistical
tests:", (untitled), "Comparing different Statistical tests & its results:" (empty
Chi-square / T-test / anova table), "links and human readable short descriptions :",
"Design and Milestones:", "Repository / Archive:", (untitled). No body text, no numbers.

### `Empirical increment EDA.docx` (2024-03-25)

Ten headings, each followed only by a chart image (no prose): "Dataset column names:"
(`student.columns` screenshot), "Target variable:" (pie: Graduate 49.93%, Dropout 32.12%,
Enrolled 17.95%, caption "Approximately 50% of students in the data have graduated."),
"Gender:" (pie: 0 = 64.83%, 1 = 35.17%), "Target variable visualization based on gender:"
(countplot, caption "According to the data, a higher number of graduates are female.
However, females also have the highest number of dropouts, although the difference
compared to males is small."), "Marital status:" (countplot, caption "Regarding marital
status, the majority of both graduates and dropouts are single."), "Course vs number of
students:" (stacked barh), "Nationality Vs number of students:" (non-Portuguese barh),
"Correlation Matrix:" (coolwarm heatmap), "Previous qualification:" (non-secondary barh),
"Age Vs number of students:" (histogram+KDE, caption "The distribution of age at enrolment
is positively skewed ... mean age at enrolment is approximately 23 years old, with the most
frequent age range falling between 19 to 25 years old."). These are the Fall 2023
notebook charts re-rendered; the captions are the Fall 2023 markdown.

### Stated limitations

No slide, notebook cell or docx in the Spring 2024 sources contains a limitations,
threats-to-validity or assumptions section. The nearest content is the "Project Future
Scope" slide quoted above and the notebooks' generic decision rule. The Final Slides deck,
which would have carried the results and discussion, contains headings only.

---

## Questions both projects asked

Deduplicated across all sources, phrased as questions. "F" = asked in Fall 2023, "S" =
asked in Spring 2024, "FS" = both.

1. FS - What share of students graduated, dropped out, or are still enrolled?
2. FS - What is the male/female split, and does gender differ between dropouts and graduates?
3. FS - Does marital status differ between dropouts and graduates? Are most students single?
4. FS - Which course has the most graduates, and which has the most dropouts?
5. FS - What is the nationality mix (how dominant are Portuguese students), and does nationality relate to dropout?
6. FS - Does previous qualification relate to dropout? Did most students enter with secondary education?
7. FS - How is age at enrollment distributed, and does mean age differ between dropouts and graduates?
8. FS - Does being a displaced student relate to dropout?
9. FS - Does being an international student relate to dropout?
10. FS - Does being a debtor relate to dropout?
11. FS - Does holding a scholarship relate to dropout?
12. FS - Do educational special needs relate to dropout?
13. F - Does having tuition fees up to date relate to dropout?
14. F - What are the most common parental occupations among graduates and dropouts?
15. FS - How is the unemployment rate distributed, and does it relate to dropout?
16. S - Does the inflation rate relate to dropout?
17. S - Does application mode relate to dropout?
18. S - Does daytime vs evening attendance relate to dropout?
19. S - Do first-semester and second-semester curricular-unit grades relate to dropout?
20. FS - Which features are correlated with each other (heatmap), and which should be dropped before modelling? (F answered: Nationality and International.)
21. S - Do mean first-semester grades differ across nationalities? across courses? across father's occupations? between debtors and non-debtors?
22. S - Do mean second-semester grades differ across mother's qualifications? across previous qualifications? across marital statuses?
23. S - Does mean age at enrollment differ across nationalities? between genders?
24. F - Can a supervised model predict Dropout vs Graduate from the enrolment-time and first-year features, and which of Gaussian NB, Logistic Regression, Random Forest, XGBoost, SVC, MLP is best?
25. F - What are accuracy, precision, recall, F1, the confusion matrix, the precision-recall curve and the ROC curve for each model?
26. F - Given one student's feature vector, what does the model predict?
27. F - Should "Enrolled" students be kept as a third class or removed? (Answered: removed for modelling; Spring merged them with Graduate as Target 0.)

---

## Numbers claimed

Every quantitative claim found, with its source, so the rebuild can check each one.

### Dataset size and composition

| Claim | Value | Source |
|---|---|---|
| Rows x columns | `(4424, 35)` | F `student.shape` (both Fall notebooks, cell 2) |
| Cells | `154840` | F `student.size` (cell 8) |
| Missing values | 0 in every column | F `student.isnull().sum()` (cell 7) |
| Duplicate rows after `drop_duplicates` | `0` | F cell 17 |
| Target counts | Graduate `2209`, Dropout `1421`, Enrolled `794` | F cell 11 |
| Target shares (pie) | Graduate 49.93%, Dropout 32.12%, Enrolled 17.95% | F cell 20 rendering; S EDA docx; S update pptx slide 7 |
| Target shares (docx pie, different rendering) | Graduate 49.9%, Dropout 32.1%, Enrolled 17.9% | F `Project description (1).docx` image 044 |
| Gender shares (pie) | 0 = 64.83%, 1 = 35.17% | F EDA notebook cell 22 rendering; S EDA docx; S update pptx slide 8 |
| "Approximately 50% of students in the data have graduated." | 50% | F cell 21 markdown; S EDA docx |
| "Variables: 35 (including target variable)"; "Samples: 4424" | 35; 4424 | S first & update pptx |
| "The dataset has 4424 data instances with 33 features and one target variable with 3 possible outcomes" | 4424; 33; 3 | F Checkpoint 1 draft docx |
| "has over 20,000 observations over 35 variables" | >20,000; 35 | F proposal PDF (contradicts the 4424 above) |
| "we have a big dataset that has 40-odd features" | ~40 | F Checkpoint 2 PDF |
| Rows after dropping Enrolled | `3630` | F cell 74 |
| Feature count for modelling | `33` | F cell 77 `(3630, 33)` |
| Train / test rows | `2904` / `726` | F cell 77 |
| Test-set class supports | class 0: `290`, class 1: `436` | F classification reports |
| Split parameters | `test_size=0.2, random_state=3` | F cell 76; docx image 051 |
| "train to test split ratio is 3:1, random state = 3" | 3:1 | F Checkpoint 2 PDF (contradicts test_size=0.2) |
| "80% of the data will be our training model and rest 20% will be the testing model. We choose the third state of the random sampling." | 80/20; random_state 3 | F cell 75 markdown; Checkpoint 1; Project description docx |
| "The mean age at enrolment is approximately 23 years old, with the most frequent age range falling between 19 to 25 years old." | 23; 19-25 | F cell 44 markdown; S EDA docx |
| "The majority of the data points in the unemployment rate distribution fall within the range of 9 to 13." | 9-13 | F cell 54 markdown |

### `student.describe().T` (F cell 9; also S pptx "Dataset Description" pictures and the per-feature captions in the F docx)

| Column | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| Marital status | 1.178571 | 0.605747 | 1.00 | 1.00 | 1.000000 | 1.000000 | 6.000000 |
| Application mode | 6.886980 | 5.298964 | 1.00 | 1.00 | 8.000000 | 12.000000 | 18.000000 |
| Application order | 1.727848 | 1.313793 | 0.00 | 1.00 | 1.000000 | 2.000000 | 9.000000 |
| Course | 9.899186 | 4.331792 | 1.00 | 6.00 | 10.000000 | 13.000000 | 17.000000 |
| Daytime/evening attendance | 0.890823 | 0.311897 | 0.00 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| Previous qualification | 2.531420 | 3.963707 | 1.00 | 1.00 | 1.000000 | 1.000000 | 17.000000 |
| Nacionality | 1.254521 | 1.748447 | 1.00 | 1.00 | 1.000000 | 1.000000 | 21.000000 |
| Mother's qualification | 12.322107 | 9.026251 | 1.00 | 2.00 | 13.000000 | 22.000000 | 29.000000 |
| Father's qualification | 16.455244 | 11.044800 | 1.00 | 3.00 | 14.000000 | 27.000000 | 34.000000 |
| Mother's occupation | 7.317812 | 3.997828 | 1.00 | 5.00 | 6.000000 | 10.000000 | 32.000000 |
| Father's occupation | 7.819168 | 4.856692 | 1.00 | 5.00 | 8.000000 | 10.000000 | 46.000000 |
| Displaced | 0.548373 | 0.497711 | 0.00 | 0.00 | 1.000000 | 1.000000 | 1.000000 |
| Educational special needs | 0.011528 | 0.106760 | 0.00 | 0.00 | 0.000000 | 0.000000 | 1.000000 |
| Debtor | 0.113698 | 0.317480 | 0.00 | 0.00 | 0.000000 | 0.000000 | 1.000000 |
| Tuition fees up to date | 0.880651 | 0.324235 | 0.00 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| Gender | 0.351718 | 0.477560 | 0.00 | 0.00 | 0.000000 | 1.000000 | 1.000000 |
| Scholarship holder | 0.248418 | 0.432144 | 0.00 | 0.00 | 0.000000 | 0.000000 | 1.000000 |
| Age at enrollment | 23.265145 | 7.587816 | 17.00 | 19.00 | 20.000000 | 25.000000 | 70.000000 |
| International | 0.024864 | 0.155729 | 0.00 | 0.00 | 0.000000 | 0.000000 | 1.000000 |
| Curricular units 1st sem (credited) | 0.709991 | 2.360507 | 0.00 | 0.00 | 0.000000 | 0.000000 | 20.000000 |
| Curricular units 1st sem (enrolled) | 6.270570 | 2.480178 | 0.00 | 5.00 | 6.000000 | 7.000000 | 26.000000 |
| Curricular units 1st sem (evaluations) | 8.299051 | 4.179106 | 0.00 | 6.00 | 8.000000 | 10.000000 | 45.000000 |
| Curricular units 1st sem (approved) | 4.706600 | 3.094238 | 0.00 | 3.00 | 5.000000 | 6.000000 | 26.000000 |
| Curricular units 1st sem (grade) | 10.640822 | 4.843663 | 0.00 | 11.00 | 12.285714 | 13.400000 | 18.875000 |
| Curricular units 1st sem (without evaluations) | 0.137658 | 0.690880 | 0.00 | 0.00 | 0.000000 | 0.000000 | 12.000000 |
| Curricular units 2nd sem (credited) | 0.541817 | 1.918546 | 0.00 | 0.00 | 0.000000 | 0.000000 | 19.000000 |
| Curricular units 2nd sem (enrolled) | 6.232143 | 2.195951 | 0.00 | 5.00 | 6.000000 | 7.000000 | 23.000000 |
| Curricular units 2nd sem (evaluations) | 8.063291 | 3.947951 | 0.00 | 6.00 | 8.000000 | 10.000000 | 33.000000 |
| Curricular units 2nd sem (approved) | 4.435805 | 3.014764 | 0.00 | 2.00 | 5.000000 | 6.000000 | 20.000000 |
| Curricular units 2nd sem (grade) | 10.230206 | 5.210808 | 0.00 | 10.75 | 12.200000 | 13.333333 | 18.571429 |
| Curricular units 2nd sem (without evaluations) | 0.150316 | 0.753774 | 0.00 | 0.00 | 0.000000 | 0.000000 | 12.000000 |
| Unemployment rate | 11.566139 | 2.663850 | 7.60 | 9.40 | 11.100000 | 13.900000 | 16.200000 |
| Inflation rate | 1.228029 | 1.382711 | -0.80 | 0.30 | 1.400000 | 2.600000 | 3.700000 |
| GDP | 0.001969 | 2.269935 | -4.06 | -1.70 | 0.320000 | 1.790000 | 3.510000 |

(count = 4424.0 for every row.) Full-precision versions of two of these appear as docx
captions: "Marital status: mean=1.1785714285714286, std=0.60574694613071";
"Age at enrollment: mean=23.265144665461122, std=7.587815615029815".

### Model metrics (Fall 2023)

| Claim | Value | Source |
|---|---|---|
| GaussianNB accuracy (= micro P/R/F1) | `0.8388429752066116`; cm `[[219 71],[46 390]]` | F notebook cell 82 |
| LogisticRegression accuracy | `0.9035812672176309`; cm `[[243 47],[23 413]]` | F cell 86 |
| RandomForestClassifier accuracy | `0.9146005509641874`; cm `[[246 44],[18 418]]` | F cell 90 |
| XGBClassifier (default) accuracy | `0.8980716253443526`; cm `[[241 49],[25 411]]` | F cell 94 |
| SVC accuracy | `0.8815426997245179`; cm `[[219 71],[15 421]]` | F cell 98 |
| MLPClassifier accuracy | `0.8966942148760331`; cm `[[241 49],[26 410]]` | F cell 101 |
| XGBClassifier(binary:logistic, n_estimators=1000) accuracy | `0.9008264462809917` | F cell 111 |
| Same XGB configuration, earlier run (repr shows `objective='multi:softprob'`) | `Accuracy: 0.7898305084745763` | F `Project description (1).docx` image 052 |
| `RandomForestClassifier(n_estimators=150, max_depth=4)` "best model" | Accuracy `0.727683615819209`; Precision `[0.79183673 0.42156863 0.79553903]`; Recall `[0.74615385 0.2654321 0.92440605]`; F1 `[0.76831683 0.32575758 0.85514486]` (P/R/F1 computed on `lr_preds`) | F `Project description (1).docx` image 053 |
| Preliminary accuracies | "Logical regression(91.75%), Random Forest(89.97%), XGBoost Classifier(90.31%), support vector classifier(91.32%)"; "average 90% accuracy" | F Checkpoint 2 PDF p.5 |
| Prediction for the "192nd record" input vector | `[1]` | F cell 113 |
| Related-work accuracies | Kiss et al. 85.3%; Magalhaes et al. 75.4% | F Checkpoint 1 draft docx |
| Per-class report values | see model table above (two-decimal values) | F cells 82-101 |

### Test statistics (Spring 2024)

All 16 chi-square, 16 t-test and 9 ANOVA statistics and p-values are tabulated in the
Spring section above; each is a printed notebook output from
`Empirical_Saieesh.ipynb` (cells 14, 18, 19, 20) and, for the subset marked, from
`Empirical_Analysis_Project_Codebook.ipynb` (cells 3-19, 22).

---

## Encodings the notebooks assumed

### Target

| Source | Mapping as written |
|---|---|
| F `Data_visualization_project.ipynb` cell 12 (and EDA notebook) | `LabelEncoder().fit_transform(student['Target'])` -> printed counts `2 2209`, `0 1421`, `1 794`, i.e. Dropout=0, Enrolled=1, Graduate=2 |
| F cell 58 markdown; Checkpoint 1; `Project description (1).docx` | "The labels dropout and graduate become 0 and 1, respectively." |
| F cell 25 / 63 | `student['Dropout'] = student['Target'].apply(lambda x: 1 if x==0 else 0)` |
| F cells 82-101 | `ConfusionMatrixDisplay(..., display_labels=['Non-Dropout', 'Dropout'])` (index 0 labelled Non-Dropout) |
| S both notebooks cell 0 | `df['Target'].map({'Graduate': 0, 'Dropout': 1, 'Enrolled': 0})` |
| S first & update pptx | "Target Variable: Student Retention (0 = dropped out, 1 = retained)" |
| F `Project description (1).docx` feature list | "Target: holds the information of Graduate, Dropout and Others. (Categorical)" |

### Categorical code -> label maps written in notebook code

Gender (F cells 27 and 23): `plt.xticks(ticks=[0,1], labels=['Female','Male'])`;
pie `labels = ['Female', 'Male']` applied to `value_counts()` order. So 0 = Female, 1 = Male.

Marital status (F cell 29): `plt.xticks(ticks=[0,1,2,3,4,5], labels=['Single','Married','Widower','Divorced','Facto Union','Legally Seperated'])` (tick positions 0-5 on a countplot whose categories are the codes 1-6, in order).

Displaced, International, Educational special needs, Debtor, Tuition fees up to date,
Scholarship holder (F cells 36-51): `plt.xticks(ticks=[0,1], labels=['No','Yes'])`, i.e. 0 = No, 1 = Yes.

Course (F cell 31), rename dict exactly as coded:
```
{1:'Biofuel Production Technologies',2:'Animation and Multimedia Design',3:'Social Service (Evening Attendance)',4:'Agronomy',5:'Communication Design',6:'Veterinary Nursing',7:'Informatics Engineering',8:'Equiniculture',9:'Management',10:'Social Service',11:'Tourism',12:'Nursing',13:'Oral Hygiene',14:'Advertising and Marketing Management',15:'Journalism and Communication',16:'Basic Education',17:'Management (Evening Attendance)'}
```

Nacionality (F cell 33):
```
{ 1:'Portuguese', 2:'German', 3:'Spanish', 4:'Italian', 5:'Dutch', 6:'English', 7:'Lithuanian', 8:'Angolan', 9:'Cape Verdean', 10:'Guinean', 11:'Mozambican', 12:'Santomean', 13:'Turkish', 14:'Brazilian', 15:'Romanian', 16:'Moldova', 17:'Mexican', 18:'Ukrainian', 19:'Russian', 20:'Cuban', 21:'Colombian'}
```

Previous qualification (F cell 40):
```
{1:'Secondary Education',2:"Higher Education—Bachelor's Degree",3:'Higher Education—Degree',4:'Higher Education—Master's Degree',5:'Higher Education—Doctorate',6:'Frequency of Higher Education',7:'12th Year of Schooling—Not Completed',8:'11th Year of Schooling—Not Completed',9:'Other—11th Year of Schooling',10:'10th Year of Schooling',11:'10th Year of Schooling—Not Completed',12:'Basic Education 3rd Cycle (9th/10th/11th year) or Equivalent',13:'Basic Education 2nd Cycle (6th/7th/8th year) or Equivalent',14:'Technological Specialization Course',15:'Higher Education—Degree (1st cycle)',16:'Professional Higher Technical Course',17:'Higher Education—Master's Degree (2nd Cycle)'}
```

Father's occupation and Mother's occupation (F cells 45 and 46, identical dict):
```
{1:'Student',2:'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',3:'Specialists in Intellectual and Scientific Activities',4:'Intermediate Level Technicians and Professions',5:'Administrative Staff',6:'Personal Services, Security and Safety Workers, and Sellers',7:'Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry',8:'Skilled Workers in Industry, Construction, and Craftsmen',9:'Installation and Machine Operators and Assembly Workers',10:'Unskilled Workers',11:'Armed Forces Professions',12:'Other Situation',13:'(blank)',14:'Armed Forces Officers',15:'Armed Forces Sergeants',16:'Other Armed Forces personnel',17:'Directors of Administrative and Commercial Services',18:'Hotel, Catering, Trade, and Other Services Directors',19:'Specialists in the Physical Sciences, Mathematics, Engineering, and Related Techniques',20:'Health Professionals',21:'Teachers',22:'Specialists in Finance, Accounting, Administrative Organization, and Public and Commercial relations',23:'Intermediate Level Science and Engineering Technicians and Professions',24:'Technicians and Professionals of Intermediate Level of Health',25:'Intermediate Level Technicians from Legal, Social, Sports, Cultural, and Similar Services',26:'Information and Communication Technology Technicians',27:'Office Workers, Secretaries in General, and Data Processing Operators',28:'Data, Accounting, Statistical, Financial Services, and Registry-related Operators',29:'Other Administrative Support Staff',30:'Personal Service Workers',31:'Sellers',32:'Personal Care workers and The Like',33:'Protection and Security Services Personnel',34:'Market-oriented Farmers and Skilled Agricultural and Animal Production Workers',35:'Farmers, Livestock Keepers, Fishermen, Hunters and Gatherers, and Subsistence',36:'Skilled Construction Workers and The Like, except Electricians',37:'Skilled Workers in Metallurgy, Metalworking, and Similar',38:'Skilled workers in Electricity and Electronics',39:'Workers in Food Processing, Woodworking, and Clothing and Other industries and Crafts',40:'Fixed Plant and Machine Operators',41:'Assembly Workers',42:'Vehicle Drivers and Mobile Equipment Operators',43:'Unskilled Workers in Agriculture, Animal Production, and Fisheries and Forestry',44:'Unskilled Workers in Extractive Industry, Construction, Manufacturing, and Transport',45:'Meal Preparation Assistants',46:'Street Vendors (except food) and Street Service Providers'}
```

Not mapped anywhere in notebook code: Application mode, Application order,
Daytime/evening attendance, Mother's qualification, Father's qualification. The Spring
2024 notebooks contain no code->label maps at all (they test on raw codes).

### Code tables embedded as images in `Project description (1).docx` ("Class Attribute entities values")

Nine images of tables headed "Table A1" to "Table A10" (the dataset's published
codebook), transcribed:

- Table A1 Marital status: 1—Single; 2—Married; 3—Widower; 4—Divorced; 5—Facto union; 6—Legally separated
- Table A2 Nationality: 1—Portuguese; 2—German; 3—Spanish; 4—Italian; 5—Dutch; 6—English; 7—Lithuanian; 8—Angolan; 9—Cape Verdean; 10—Guinean; 11—Mozambican; 12—Santomean; 13—Turkish; 14—Brazilian; 15—Romanian; 16—Moldova (Republic of); 17—Mexican; 18—Ukrainian; 19—Russian; 20—Cuban; 21—Colombian
- Table A3 Application mode: 1—1st phase—general contingent; 2—Ordinance No. 612/93; 3—1st phase—special contingent (Azores Island); 4—Holders of other higher courses; 5—Ordinance No. 854-B/99; 6—International student (bachelor); 7—1st phase—special contingent (Madeira Island); 8—2nd phase—general contingent; 9—3rd phase—general contingent; 10—Ordinance No. 533-A/99, item b2) (Different Plan); 11—Ordinance No. 533-A/99, item b3 (Other Institution); 12—Over 23 years old; 13—Transfer; 14—Change in course; 15—Technological specialization diploma holders; 16—Change in institution/course; 17—Short cycle diploma holders; 18—Change in institution/course (International)
- Table A4 Course: 1—Biofuel Production Technologies; 2—Animation and Multimedia Design; 3—Social Service (evening attendance); 4—Agronomy; 5—Communication Design; 6—Veterinary Nursing; 7—Informatics Engineering; 8—Equiniculture; 9—Management; 10—Social Service; 11—Tourism; 12—Nursing; 13—Oral Hygiene; 14—Advertising and Marketing Management; 15—Journalism and Communication; 16—Basic Education; 17—Management (evening attendance)
- Table A5 Previous qualification: 1—Secondary education; 2—Higher education—bachelor's degree; 3—Higher education—degree; 4—Higher education—master's degree; 5—Higher education—doctorate; 6—Frequency of higher education; 7—12th year of schooling—not completed; 8—11th year of schooling—not completed; 9—Other—11th year of schooling; 10—10th year of schooling; 11—10th year of schooling—not completed; 12—Basic education 3rd cycle (9th/10th/11th year) or equivalent; 13—Basic education 2nd cycle (6th/7th/8th year) or equivalent; 14—Technological specialization course; 15—Higher education—degree (1st cycle); 16—Professional higher technical course; 17—Higher education—master's degree (2nd cycle)
- Table A6 Mother's qualification / Father's qualification: 1—Secondary Education—12th Year of Schooling or Equivalent; 2—Higher Education—bachelor's degree; 3—Higher Education—degree; 4—Higher Education—master's degree; 5—Higher Education—doctorate; 6—Frequency of Higher Education; 7—12th Year of Schooling—not completed; 8—11th Year of Schooling—not completed; 9—7th Year (Old); 10—Other—11th Year of Schooling; 11—2nd year complementary high school course; 12—10th Year of Schooling; 13—General commerce course; 14—Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent; 15—Complementary High School Course; 16—Technical-professional course; 17—Complementary High School Course—not concluded; 18—7th year of schooling; 19—2nd cycle of the general high school course; 20—9th Year of Schooling—not completed; 21—8th year of schooling; 22—General Course of Administration and Commerce; 23—Supplementary Accounting and Administration; 24—Unknown; 25—Cannot read or write; 26—Can read without having a 4th year of schooling; 27—Basic education 1st cycle (4th/5th year) or equivalent; 28—Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent; 29—Technological specialization course; 30—Higher education—degree (1st cycle); 31—Specialized higher studies course; 32—Professional higher technical course; 33—Higher Education—master's degree (2nd cycle); 34—Higher Education—doctorate (3rd cycle)
- Table A7 Mother's occupation / Father's occupation: codes 1-46 with the same labels as the notebook dict above (12—Other Situation; 13—(blank))
- Table A8 Gender: 1—male; 0—female
- Table A9 Daytime/evening attendance: 1—daytime; 0—evening
- Table A10 Yes/No attributes (Displaced, Educational special needs, Debtor, Tuition fees up to date, Scholarship holder, International): 1—yes; 0—no

Observed code ranges in the data (from `describe()` max): Application mode 18, Course 17,
Previous qualification 17, Nacionality 21, Mother's qualification 29, Father's
qualification 34, Mother's occupation 32, Father's occupation 46, Marital status 6,
Application order 0-9.

### Feature descriptions as written in `Project description (1).docx` ("Task 2")

Type labels assigned there: Categorical - Marital status, Application mode, Course,
Daytime/evening attendance, Previous qualification, Nacionality, Mother's qualification,
Father's qualification, Mother's occupation, Father's occupation, Displaced, Educational
special needs, Debtor, Tuition fees up to date, Gender, Scholarship holder, International,
Target. Numerical - Application order, Age at enrollment, all twelve "Curricular units"
columns, Unemployment Rate, Inflation Rate, GDP. The grade columns are described as "The
number of curricular units grade received the student in the first semester"
(sic; the data holds a float average grade 0-18.875).
