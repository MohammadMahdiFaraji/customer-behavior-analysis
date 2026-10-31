# Customer behavior analysis (Machine Learning course, project 1)

**Author:** MohammadMahdi Faraji
**Course:** Machine Learning, Prof. Hamed Baghban, fall 2024. This was the first project of the course. After the course I did it again, more carefully, as a worked example for people who are new to machine learning.

The data is a table of 1000 customers of an online shop. For 700 of them the purchase category is missing. The project asked to fill in the missing categories, check if the categories are balanced, evaluate a classifier with a confusion matrix and F1 score, find the valuable customers, and explain how an unbalanced distribution affects accuracy. The main lesson for me was that a model with 70 % accuracy can still be useless.

<p align="center"><img src="results/figures/confusion_matrices.png" width="95%"></p>
<p align="center"><sub>Rows are the real categories, columns are the predictions. Left: a rule that always predicts Electronics. Middle: random forest. Right: random forest trained with SMOTE.</sub></p>

## The idea in simple words

Think of a student who answers "true" to every question of a true/false exam. If 65 % of the answers are "true", this student gets 65 % without reading the questions. My first version of this project had the same problem. The random forest predicted "Electronics" for all customers, and because about 65 % of the customers bought electronics, it got 70 % accuracy on my small test set. The model had not learned anything.

In this new version I answer the six tasks of the project ([`docs/assignment.md`](docs/assignment.md)) and pay more attention to three points:
- why the values are missing, and why filling them with a model can be a bad idea,
- why accuracy is not enough when the classes are unbalanced,
- comparing every model with a very simple baseline.

## At a glance

| | |
|---|---|
| Data | 1000 customers (synthetic data given in the course), 12 columns |
| Missing values | purchase category of 700 customers, missing at random |
| Known categories (300) | Electronics 65 %, Clothing 17 %, Groceries 12 %, Books 7 % |
| Methods | correlation, statistical tests, k-means, logistic regression, decision tree, random forest, class weights, oversampling, SMOTE, cross-validation |
| Main result | no model is clearly better than simple guessing: best accuracy 0.65 (the same as always predicting Electronics), best macro F1 0.24 (random guessing: 0.23) |
| Valuable customers | the top 20 % by spending bring 42 % of the money |
| Run time | about 2 minutes, in Colab or on a PC |

## Results

### 1. The category can't be predicted from these columns

I evaluated each model on the 300 customers whose category is known, with 5-fold cross-validation repeated 5 times (25 tests per model). I also added two simple baselines that don't use the customer data at all.

| | accuracy | balanced accuracy | macro F1 |
|---|---:|---:|---:|
| always "Electronics" | 0.65 | 0.25 | 0.20 |
| random guess with the known shares | 0.45 | 0.23 | 0.23 |
| logistic regression | 0.64 | 0.26 | 0.21 |
| random forest | 0.64 | 0.25 | 0.20 |
| random forest with SMOTE | 0.52 | 0.25 | 0.24 |
| logistic regression with class weights | 0.22 | 0.25 | 0.19 |

All 8 models and the 2 baselines are in [`results/model_comparison.csv`](results/model_comparison.csv). What the table shows:
- No model has a higher accuracy than always predicting Electronics.
- Balanced accuracy (the average recall of the four categories) is about 0.25 for all models. Random guessing between four categories also gives 0.25.
- Macro F1 (the average F1 of the four categories) is 0.24 for the best model and 0.23 for random guessing. This difference is smaller than the change from one test to another (about 0.05).
- I also ran a permutation test: the model was trained 200 times on shuffled categories. The p-value is 0.40, so the model does not do better on the real categories than on random ones.

### 2. Filling the missing values with a model is misleading

A random forest trained on the 300 known customers predicts "Electronics" for 687 of the 700 missing ones. After this filling, 88 % of the table is Electronics instead of 65 %. A new model tested on the filled table gets 0.88 accuracy, but this only shows that two models agree with each other. This is called data leakage. So I filled the missing values with random draws that keep the known shares, marked them in a new column `category_filled`, and used only the real categories for testing.

### 3. Valuable customers

The median customer spends about 84. The top 20 % spend at least 147, and together they bring 42 % of all the money. Apart from spending, they are very similar to the other customers in age, gender, region, time on the website and category. They only view a few more products (10.7 against 9.7). So in this data the valuable customers can only be found by how much they spend.

### 4. Effect of an unbalanced distribution on accuracy

Our data has no signal, so to show this effect I made a simple two-class dataset with `make_classification` and made the small class rarer step by step:

<p align="center"><img src="results/figures/imbalance_effect.png" width="95%"></p>

Without class weights (left), the accuracy goes up from 0.70 to 0.97, but the recall of the small class goes down from 0.65 to 0.05. At 98/2 the model almost always predicts the big class. With class weights (right), the model still finds about half of the small class at 98/2, but the accuracy is lower.

## How it works, step by step

0. **Missing values.** I compared the 300 customers with a category and the 700 without one. Their averages are similar in every column (all p-values above 0.1), and a classifier can't separate the two groups (ROC AUC 0.50). So the values are missing at random, and the 300 known rows are a fair sample.
1. **Correlation (task 1).** Only `Feature_1` to `Feature_5` are correlated with each other (strongest pair *r* = -0.80). In a scatter plot they form four separate groups of 250 customers, which looks like the output of a random data generator. None of the columns is clearly related to the category. I didn't remove any column, because no column is a copy of another one and the spending columns are needed for task 5. (In my first version I removed `Purchase_Amount`, which task 5 needs.)
2. **Filling the missing categories (task 2).** I compared three ways: the most common category, a model, and random draws with the known shares. I used the random draws and added the column `category_filled`. The completed table is [`results/completed_dataset.csv`](results/completed_dataset.csv).
3. **Balance (task 3).** The bar chart shows about ten Electronics buyers for each book buyer. The notebook explains class weights, oversampling, SMOTE and undersampling, and shows SMOTE on the known rows. Balancing is done only on the training part of each fold, never on the test part.
4. **Confusion matrix and F1 (task 4).** Logistic regression, decision tree and random forest, plain and with class weights, oversampling or SMOTE (8 models), compared with the 2 baselines. Accuracy, balanced accuracy, macro F1, weighted F1, confusion matrices and a permutation test.
5. **Customer behavior (task 5).** The spending distribution, the top 20 %, a comparison of the top customers with the others, and four customer groups from k-means on spending, time on the website and viewed products.
6. **Effect of imbalance (task 6).** Numbers from our data, and the experiment above.

## Mistakes in my first version

The original notebook is in [`legacy/original_course_submission/`](legacy/original_course_submission/), with a list of its problems. The main ones:
- It ran only because of variables left from earlier runs. In a new session it stops with a `NameError`.
- The confusion matrix titled "Decision Tree" was actually from the random forest.
- The random forest and the neural network predicted "Electronics" for every test customer, and the weighted F1 did not show this.
- The random forest filled the 700 missing values, so 883 of the 1000 customers became Electronics buyers.
- The written discussion and tasks 5 and 6 were missing.

More details are in [`docs/changes_from_course_project.md`](docs/changes_from_course_project.md).

## Limitations

- The data is synthetic, so the results say little about real customers.
- There are only 20 book buyers, so every number about books is uncertain.
- `Feature_3` and gender show a weak relation with the category (p = 0.006 and 0.015). But I tested 12 columns, and with 12 tests the chance that at least one p-value is under 0.05 just by luck is about 46 %. New data would be needed to check it.

## How to run

**In Colab:** open `notebooks/customer_behavior_analysis.ipynb` in Google Colab and choose *Runtime > Run all*. The notebook asks you to upload `data/synthetic_customer_behavior.csv`. If `imbalanced-learn` is not installed, the first cell installs it.

**On your PC** (tested with Python 3.14):
```
pip install -r requirements.txt
cd notebooks
jupyter notebook customer_behavior_analysis.ipynb
```
After running all cells, the figures and tables are saved in `results/`. To check the main numbers again with plain numpy (without pandas or scikit-learn):
```
python tools/check_numbers.py
```

## Files

```
├── notebooks/customer_behavior_analysis.ipynb   the whole analysis, task by task, with explanations
├── data/synthetic_customer_behavior.csv         the data from the course
├── results/                                     figures, the model table, the completed data, a summary
├── tools/check_numbers.py                       independent check of the main numbers
├── docs/                                        the project description in my own words, what changed since the course
└── legacy/original_course_submission/           my original project, unchanged except for my student ID
```

## Thanks

- Prof. Hamed Baghban, for the course and the project.
- I used an AI assistant to help review and rewrite the code, to clean it up and prepare it for publishing.

## License

MIT, see [LICENSE](LICENSE).
