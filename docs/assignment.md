# The assignment

Homework 1 of the Machine Learning course (Prof. Hamed Baghban), first semester of the 2024-2025 school year. The original sheet was in Persian; this is the content in my own words.

We got a table with information about the customers of an online shop. For many customers the column `Purchase_Category` (what kind of product they bought) is empty. The tasks were:

1. Compute the correlation between the features, and remove features that are not needed.
2. Fill in the missing values of `Purchase_Category`.
3. Draw a bar chart of `Purchase_Category` and discuss whether the classes are balanced. If they are not, suggest a way to make the distribution more balanced, and do it.
4. For the data from the previous steps, compute the confusion matrix and the F1 score.
5. Use the results to describe how the customers behave (for example: which customers are valuable?).
6. Explain how an unbalanced distribution of one feature affects the accuracy of the results.

Everything (code, the chosen methods and the reasons for them, plots and the discussion of the results) had to be handed in as one Jupyter notebook.

The new notebook answers the tasks in this order: [`notebooks/customer_behavior_analysis.ipynb`](../notebooks/customer_behavior_analysis.ipynb).
