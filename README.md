## Decision Tree from Scratch

This repository contains my full implementation of a Decision Tree classifier built completely from scratch in Python, without using sklearn.tree.

The main purpose of this project was to truly understand how decision trees work internally — not just use them as black-box models. I implemented the splitting logic, impurity calculations, and recursive tree construction myself to see what really happens behind the scenes.

The dataset used in this project is the Bank Marketing dataset (bank-additional-full.csv).

## 📂 Project Structure

```bash

decision-tree-from-scratch/
│
├── DT.ipynb                    # Step-by-step development and experiments
├── finall.py                   # Final runnable script
├── bank-additional-full.csv    # Dataset
├── output-depth3.png           # Example tree visualization
└── README.md
```
## DT.ipynb — Step-by-Step Development

In the notebook, the model was developed gradually.

We:

Implemented Gini Impurity and Entropy manually

Calculated Information Gain

Built the recursive tree construction logic

Tested different stopping conditions

Tuned hyperparameters such as:

max_depth

min_samples_split

min_samples_leaf

criterion (gini or information gain)

The notebook also includes experimentation with different depth values and comparison between training and validation accuracy to better understand overfitting behavior.

This part of the project focuses more on learning and experimentation.

## finall.py — One-Click Full Model

The finall.py file is the clean and final version of the implementation.

With a single run:

python finall.py


The script:

Loads and preprocesses the dataset

Trains the decision tree

Prints the full tree structure in the console

Reports training, validation, and test accuracy

Generates a visualization of the tree

This file is designed to be straightforward and easy to execute without going through the notebook.

## What’s Implemented

The decision tree includes:

Manual Gini Impurity calculation

Manual Entropy / Information Gain calculation

Exhaustive feature split search

Recursive tree building

Custom stopping conditions

Handling of edge cases (pure nodes, depth limits, small splits)

No external machine learning model is used — all splitting logic is implemented manually.

scikit-learn is used only for dataset splitting, not for modeling.

## Data Preprocessing

Categorical features are encoded manually

Some continuous features are discretized (binned)

Dataset is split into train, validation, and test sets

## Tree Output
Console Output

The tree structure is printed using a custom print_tree() function.

This shows:

The feature used at each split

The gain or impurity value

Class distribution at each node

This makes the model fully interpretable and easy to debug.

PNG Visualization

An example tree visualization is provided in:

output-depth3.png


This shows the generated tree structure for a depth-limited model.

## ▶️ How to Run
git clone https://github.com/setayeshbaghaee/decision-tree-from-scratch.git
cd decision-tree-from-scratch

pip install pandas matplotlib scikit-learn
python finall.py


After running:

The decision tree structure appears in the terminal

Accuracy metrics are displayed

A tree visualization image is generated

## What I Learned

Working on this project helped me better understand:

How impurity metrics actually guide split selection

How recursive tree construction works internally

How hyperparameters affect model complexity and overfitting

The difference between theoretical algorithm definitions and real implementation challenges

Building the algorithm from scratch made concepts like Information Gain and stopping criteria much clearer compared to using ready-made libraries.

## Possible Improvements

Implement threshold-based splits for continuous features (instead of binning)

Add feature importance calculation

Add confusion matrix and F1-score reporting

Export tree structure to Graphviz for cleaner visualization

Optimize split search for larger datasets


## 📌 Author

Setayesh Baghaee

Computer Engineering Student
