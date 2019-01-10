# Home Credit Loan Default Prediction

Kaggle's <a href="https://www.kaggle.com/c/home-credit-default-risk">Home Credit Default Risk</a> contest featured a $70,000 dollar prize for the most accurate model predicting which loan applications will result in the applicant defaulting on the loan (with accuracy measured by area under the ROC curve). 

This project (full content in the <a href="https://github.com/gqfiddler/predicting-loan-defaults/blob/master/Home%20Credit%20Default%20Prediction.ipynb">"Home Credit Default Prediction.ipynb"</a> file above) details the entire data handling and model experimentation process. Particular techniques demonstrated include:
- data cleaning and visualization
- merging disparate data tables
- benchmarking multiple model types
- measuring and visualizing feature importances
- GLRM-based feature reduction
- extensive feature engineering
- different methods of handling categorical data

## Results

My final AUROC score, with an extensively feature-engineered LightGBM gradient boosted tree model, was 0.789, less than two percentage points from the winning score of 0.806.




