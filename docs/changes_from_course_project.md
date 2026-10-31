# What changed since the course version

The course version is in [`legacy/original_course_submission/`](../legacy/original_course_submission/). Its problems, with the place where each one is, are listed in the [README of that folder](../legacy/original_course_submission/README.md). This page goes through the six tasks and says what the first version did and what the new notebook does instead.

## Running the notebook

| First version | New version |
|---|---|
| Worked only in my Colab session, because some variables (`rf_pred`, `numerical_columns`) were left over from earlier runs | Runs from top to bottom in a fresh session, in Colab or locally, in about 2 minutes |
| The decision tree and the neural network had no fixed random seed | One seed (`SEED = 42`) everywhere; two runs give exactly the same numbers, tables and plots |
| No saved results | Plots, tables, the completed data and a summary are written to `results/`, and `tools/check_numbers.py` checks the main numbers with plain numpy |

## Task by task

| Task | First version | New version |
|---|---|---|
| Before starting | Did not ask why the categories are missing | Checks it: the 300 known and 700 missing rows look the same in every column (all p > 0.1, ROC AUC 0.50), so the gaps are random |
| 1. Correlation | Pearson correlation on the 300 known rows only, with gender, region and category turned into the numbers 0 to 3. `Feature_1`, `Feature_5` and `Purchase_Amount` dropped without a reason | Pearson correlation of the number columns on all 1000 rows; ANOVA and chi-square / Cramér's V for the connection with the category. Finds that `Feature_1..5` are four artificial clouds of 250 customers. Keeps all columns and explains why |
| 2. Missing categories | Filled with the random forest: 689 of 700 became Electronics | Compares the mode, a model and random draws with the known shares; keeps the random draws and marks every filled value in `category_filled`. Only real categories are used for testing |
| 3. Balance | A bar chart of the filled data with the classes shown as 0.0 to 3.0; no discussion, no balancing | Bar charts with names and percentages (known data, filled by a model, filled by random draws); class weights, oversampling, SMOTE and undersampling explained; balancing done only on the training folds |
| 4. Confusion matrix and F1 | One 80/20 split (60 test rows, 1 book buyer); the decision tree's plot was really the random forest's; only weighted F1 | Repeated stratified 5-fold cross-validation (25 tests); two simple baselines; accuracy, balanced accuracy, macro and weighted F1; confusion matrices with names; a permutation test; a demonstration of data leakage |
| 5. Customer behavior | KMeans on unscaled columns and a scatter plot, without a conclusion | Spending distribution, the top 20 % and their share of the money, a comparison of big spenders with the others, behavior segments on scaled columns, and a written answer |
| 6. Effect of imbalance | Missing | Our own numbers plus a controlled experiment (50/50 up to 98/2) with and without class weights |
| Neural network | Unscaled inputs, validation taken from the test set, predicted the same class for every customer | Not repeated: the assignment did not ask for it, and the simpler models already show that the columns hold no information about the category |

## Results, before and after

| | First version | New version |
|---|---|---|
| Reported accuracy | 0.70 (random forest), 0.65 (decision tree) | 0.65 for "always Electronics"; no model is better |
| Reported F1 | weighted F1 0.58 and 0.55 | macro F1 about 0.20 to 0.24, the same as the baselines |
| Filled data | 883 of 1000 Electronics | 639 of 1000 Electronics (the known share), and the 700 filled rows are marked |
| Conclusion | none written | the category can't be predicted from these columns; valuable customers are the top 20 % by spending (42 % of the money) |
