# My original project (December 2024)

This is the notebook I handed in for project 1 of the Machine Learning course (Prof. Hamed Baghban). I kept it unchanged, with the outputs of my Colab run, so that the problems listed below can be checked. The only change is that I removed my student ID from the first cell and from the file name.

- `MohammadMahdi_Faraji_project1.ipynb`: the notebook.
- The data file it reads, `synthetic_customer_behavior (1).csv`, is the same file as [`data/synthetic_customer_behavior.csv`](../../data/synthetic_customer_behavior.csv) (byte for byte, only the name is shorter).
- The project description is not included. Its content is in [`docs/assignment.md`](../../docs/assignment.md).

## Does it still run?

Not from top to bottom. In my Colab session the cells had been run before (the first code cell has the number 121), so some variables already existed:

- The decision tree cell uses `rf_pred`, which is only made in the random forest cell after it. In a fresh session this is a `NameError`.
- The KMeans cell uses `numerical_columns`, which is not defined anywhere in the notebook.
- With pandas 3, writing the predicted numbers into the text column `Purchase_Category` fails with a `TypeError`. It worked with the pandas 2 of Colab at that time.

When the random forest cell is run before the decision tree cell, `numerical_columns` is set to all number columns, pandas is told to behave like version 2 and the Keras cell is skipped, the printed numbers come out exactly the same as in the notebook: decision tree accuracy 0.65 and F1 0.5515, random forest accuracy 0.70 and F1 0.5765.

## Where the problems are

| Problem | Where to find it |
|---|---|
| The notebook depends on old variables (`rf_pred`, `numerical_columns`) and does not run in a fresh session | "Decision Tree Classifier" and "plot the results" cells |
| The confusion matrix under "Decision Tree" is the random forest's: it uses `rf_pred` and the title says "Random Forest". The two plots are identical | "Decision Tree Classifier" cell |
| The random forest predicts "Electronics" for all 60 test rows. Its 70 % is what you get by always saying "Electronics" (42 of the 60 test rows are Electronics). The weighted F1 hides this; the macro F1 would be about 0.21 | output of the "random forest" cell, and its confusion matrix |
| The neural network also predicts the same class (2, Electronics) for every test row. Its inputs were not scaled (`StandardScaler` is imported but never used) | last cell |
| The 700 missing categories were filled with the random forest's predictions: 689 became Electronics. After that, 883 of the 1000 customers are Electronics, so filling made the imbalance much worse | "random forest" and "plot the results" cells |
| No discussion of balance and no balancing method, although task 3 asks for both | missing |
| The correlation was computed only on the 300 rows with a known category, and the text columns (gender, region, category) were turned into the numbers 0, 1, 2, 3 and put into a Pearson correlation, which does not make sense for them | "Encode categorical features" cell |
| `Feature_1`, `Feature_5` and `Purchase_Amount` were dropped without a reason. `Purchase_Amount` is the column that task 5 (valuable customers) needs most | "Select features and target" cell |
| The encoder was fitted a second time on the rows without a category. This only gives the same numbers because both parts happen to contain the same genders and regions | "Encode categorical features" cell |
| KMeans was run on columns with very different scales, so the clusters do not follow the four clear groups in the plot | "plot the results" cell |
| The classes in the final bar chart are shown as 0.0 to 3.0 instead of their names | "plot the results" cell |
| Tasks 5 and 6, and the written reasons and discussion that the assignment asked for, are missing | whole notebook |

The [main README](../../README.md) and [`docs/changes_from_course_project.md`](../../docs/changes_from_course_project.md) describe what the new notebook does instead.
