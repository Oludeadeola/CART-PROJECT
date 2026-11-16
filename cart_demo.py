"""
================================================================================
 CART (Classification and Regression Trees) - A COMPLETE CLASSROOM DEMO
================================================================================
 What is CART?
   CART is the algorithm behind scikit-learn's decision trees. It builds a
   BINARY tree: at every node it asks one yes/no question about ONE feature
   (e.g. "petal length <= 2.45 cm?") and splits the data into two groups.
   It keeps splitting until the groups are pure enough or a stopping rule fires.

   - For CLASSIFICATION the split is chosen to reduce impurity (Gini or Entropy).
   - For REGRESSION the split is chosen to reduce squared error (variance).

 How to run:
   Google Colab : paste into a cell and press Run. Nothing to install.
   VS Code      : python cart_demo.py
                  (needs: pip install pandas numpy matplotlib seaborn scikit-learn)

 Every chart opens in its own window/output cell. In VS Code the script pauses
 on each chart until you close it - that is normal.
================================================================================
"""

# =============================================================================
# SECTION 0 - IMPORTS AND GLOBAL PLOT SETTINGS
# -----------------------------------------------------------------------------
# WHAT: load every library we need, once, at the top.
# WHY : keeping imports together makes the script easy to read, and it is the
#       standard style used in real data-science projects.
# =============================================================================

import numpy as np                      # numerical arrays and maths
import pandas as pd                     # tables (DataFrames) - nice printing
import matplotlib.pyplot as plt         # the plotting engine
import seaborn as sns                   # prettier statistical plots (heatmap)

from sklearn.datasets import load_iris, load_diabetes          # built-in datasets
from sklearn.model_selection import train_test_split           # train/test split
from sklearn.tree import (                                     # the CART models
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,                                                 # tree drawing
)
from sklearn.metrics import (                                  # scoring tools
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)

# --- Plot style -------------------------------------------------------------
# A fixed, colour-blind-safe pair of colours used across the whole demo:
# BLUE always means "training data", ORANGE always means "testing data".
# Keeping the meaning of a colour constant is a basic data-visualisation rule.
COLOR_TRAIN = "#2A6FB0"     # blue  -> training set
COLOR_TEST = "#D9822B"      # orange-> testing set
sns.set_theme(style="whitegrid")        # light grid background for all charts
plt.rcParams["figure.dpi"] = 110        # slightly sharper figures
plt.rcParams["axes.titlesize"] = 13     # readable titles when projected


def show(title_for_console):
    """Small helper: tidy the layout, print a marker, then display the figure.

    WHY: during a live presentation it helps to see in the console which chart
    is on screen right now."""
    plt.tight_layout()                  # stops titles/labels overlapping
    print(f"   [chart] {title_for_console}")
    plt.show()                          # draw it


print("\n" + "=" * 78)
print("PART 1 - CART FOR CLASSIFICATION (Iris dataset)")
print("=" * 78)

# =============================================================================
# SECTION 1.1 - LOAD THE IRIS DATA AND LOOK AT IT
# -----------------------------------------------------------------------------
# WHAT: Iris contains 150 flowers, 4 measurements each, and 3 species.
# WHY : it is small, clean and famous, so the class can focus on the ALGORITHM
#       instead of on data cleaning.
# =============================================================================

iris = load_iris()                              # a dictionary-like "Bunch" object

# Build a DataFrame so the data looks like a spreadsheet (easier to explain).
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)   # the 4 features
iris_df["species"] = iris.target                                # 0, 1, 2 codes
# Add a human-readable name column purely for display/teaching.
iris_df["species_name"] = iris_df["species"].map(
    dict(enumerate(iris.target_names))          # {0:'setosa', 1:'versicolor', ...}
)

print("\nFirst 5 rows of the Iris dataset:")
print(iris_df.head())                           # requirement: show head()
print(f"\nShape (rows, columns): {iris_df.shape}")
print(f"Class names           : {list(iris.target_names)}")
print("Class counts          :")
print(iris_df["species_name"].value_counts())   # 50 of each - a balanced dataset

# =============================================================================
# SECTION 1.2 - SEPARATE FEATURES (X) AND TARGET (y)
# -----------------------------------------------------------------------------
# WHAT: X = the inputs the tree is allowed to ask questions about.
#       y = the answer we want it to predict.
# WHY : every supervised-learning model in scikit-learn expects this split.
# =============================================================================

X = iris_df[iris.feature_names]      # the 4 measurement columns
y = iris_df["species"]               # the numeric class label (0/1/2)

print(f"\nX shape: {X.shape}   (150 flowers x 4 features)")
print(f"y shape: {y.shape}   (150 labels)")

# =============================================================================
# SECTION 1.3 - TRAIN / TEST SPLIT (80% / 20%)
# -----------------------------------------------------------------------------
# WHAT: hide 20% of the flowers from the model and use them only for marking.
# WHY : a tree can memorise the data it was trained on, so training accuracy is
#       NOT evidence that it learned anything. The test set is the honest exam.
# NOTE: stratify=y keeps the 3 species balanced inside both halves;
#       random_state=42 makes the split reproducible - everyone gets the same
#       numbers, which matters when the class runs the code themselves.
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20% goes to testing -> 30 flowers
    random_state=42,        # fixed seed = same split every run
    stratify=y,             # keep class proportions equal in both parts
)

print(f"\nTraining set: {X_train.shape[0]} flowers")
print(f"Testing  set: {X_test.shape[0]} flowers")

# =============================================================================
# SECTION 1.4 - TRAIN THE CART CLASSIFIER (GINI)
# -----------------------------------------------------------------------------
# WHAT: fit a decision tree that uses the GINI impurity measure.
# WHY : Gini measures how "mixed" a group of flowers is.
#         Gini = 1 - sum(p_i^2)
#       0.0  = perfectly pure node (all flowers same species)
#       ~0.67 = worst case with 3 equally mixed classes.
#       CART tries every feature and every threshold and keeps the split that
#       lowers the weighted Gini the most. That is the whole training loop.
# NOTE: no max_depth here, so the tree grows until the leaves are pure -> this
#       is the "fully grown" tree we will later prune.
# =============================================================================

clf_gini = DecisionTreeClassifier(
    criterion="gini",       # the splitting rule
    random_state=42,        # tie-breaking is random; fix it for reproducibility
)
clf_gini.fit(X_train, y_train)          # <- this single line IS the CART algorithm

# =============================================================================
# SECTION 1.5 - PREDICT AND SCORE
# -----------------------------------------------------------------------------
# WHAT: run the 30 unseen flowers down the tree and compare to the truth.
# WHY : accuracy = "what fraction did we get right"; the classification report
#       adds precision / recall / f1 PER SPECIES, which exposes problems that a
#       single accuracy number can hide.
# =============================================================================

y_pred = clf_gini.predict(X_test)                    # predicted species codes
acc_gini = accuracy_score(y_test, y_pred)            # overall accuracy 0..1

print(f"\nTest accuracy (Gini tree): {acc_gini:.4f}  ({acc_gini * 100:.2f}%)")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
# Reading the report:
#   precision = of the flowers I CALLED setosa, how many really were setosa?
#   recall    = of the flowers that really ARE setosa, how many did I catch?
#   f1-score  = the balance between the two.
#   support   = how many test flowers of that species exist (here only ~10 each!)

# =============================================================================
# SECTION 1.6 - CONFUSION MATRIX HEATMAP
# -----------------------------------------------------------------------------
# WHAT: a grid - rows are the TRUE species, columns are the PREDICTED species.
# WHY : the diagonal is the correct answers; anything off the diagonal tells you
#       exactly WHICH species the model mixes up (usually versicolor/virginica).
# =============================================================================

cm = confusion_matrix(y_test, y_pred)                # 3x3 count grid

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,                      # write the count inside each cell
    fmt="d",                         # as whole numbers
    cmap="Blues",                    # single-hue sequential ramp: light = few
    cbar_kws={"label": "Number of flowers"},
    linewidths=2,                    # thin gap between cells keeps them distinct
    linecolor="white",
    xticklabels=iris.target_names,   # readable labels instead of 0/1/2
    yticklabels=iris.target_names,
)
plt.title("Confusion Matrix - CART (Gini) on Iris Test Set")
plt.xlabel("Predicted species")
plt.ylabel("True species")
show("Confusion matrix")

# =============================================================================
# SECTION 1.7 - DRAW THE WHOLE TREE
# -----------------------------------------------------------------------------
# WHAT: the actual rules the model learned, as a picture.
# WHY : THIS is why CART is loved - unlike a neural network, you can read the
#       model out loud. Each box shows:
#         line 1 : the question  (feature <= threshold)
#         gini   : impurity of that node
#         samples: how many training flowers reached that box
#         value  : how they split across the 3 species
#         class  : the majority species = what a leaf would predict
#       filled=True colours each box by its majority class; stronger colour
#       means a purer node.
# =============================================================================

plt.figure(figsize=(16, 9))
plot_tree(
    clf_gini,
    feature_names=iris.feature_names,       # show real names, not "X[2]"
    class_names=list(iris.target_names),    # show species, not 0/1/2
    filled=True,                            # colour by majority class
    rounded=True,                           # softer boxes, easier to read
    fontsize=9,
)
plt.title("Full CART Decision Tree (criterion = gini) - Iris")
show("Full decision tree")

# =============================================================================
# SECTION 1.8 - TREE SIZE
# -----------------------------------------------------------------------------
# WHAT: depth = longest chain of questions; leaves = number of final decisions.
# WHY : these two numbers ARE the complexity of the model. A deeper tree with
#       more leaves fits the training data better but generalises worse.
# =============================================================================

print(f"\nTree depth            : {clf_gini.get_depth()}")
print(f"Number of leaves      : {clf_gini.get_n_leaves()}")
print(f"Total nodes           : {clf_gini.tree_.node_count}")

# =============================================================================
# SECTION 1.9 - FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
# WHAT: how much total impurity each feature removed, scaled to sum to 1.
# WHY : it answers "which measurement actually decides the species?" For Iris
#       the petal measurements do nearly all the work; sepal width is almost
#       useless. This is a free form of feature selection that trees give you.
# =============================================================================

importances = pd.Series(
    clf_gini.feature_importances_,      # one number per feature, sums to 1.0
    index=iris.feature_names,
).sort_values()                         # sorted -> the bar chart reads top-down

print("\nFeature importances (higher = more useful for splitting):")
print(importances.sort_values(ascending=False))

plt.figure(figsize=(8, 4.5))
bars = plt.barh(
    importances.index,                  # horizontal bars: long names stay readable
    importances.values,
    color=COLOR_TRAIN,                  # one series -> one colour, no legend needed
    height=0.6,                         # thin marks, gap between bars
)
# Direct value labels: the reader should not have to measure against the axis.
for bar, value in zip(bars, importances.values):
    plt.text(value + 0.012, bar.get_y() + bar.get_height() / 2,
             f"{value:.3f}", va="center", fontsize=9, color="#333333")
plt.title("Which Measurements Does the Tree Actually Use? (Gini importance)")
plt.xlabel("Importance (share of total impurity reduction)")
plt.ylabel("Feature")
plt.xlim(0, max(importances.values) * 1.18)     # headroom so labels fit
show("Feature importance")


print("\n" + "=" * 78)
print("PART 2 - GINI vs ENTROPY")
print("=" * 78)

# =============================================================================
# SECTION 2 - THE SECOND SPLITTING CRITERION
# -----------------------------------------------------------------------------
# WHAT: train the same tree again but score splits with ENTROPY instead.
#         Entropy = -sum(p_i * log2(p_i))      (0 = pure, 1.585 = 3-way mixed)
#       "Information gain" = entropy before the split minus entropy after.
# WHY : students always ask which is better. The honest answer is: almost
#       always the same. Entropy uses a logarithm so it punishes mixed nodes
#       slightly harder and can produce marginally different trees, but in
#       practice the accuracy difference is noise. Gini is the default because
#       it is cheaper to compute (no log).
# =============================================================================

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf_entropy.fit(X_train, y_train)

acc_entropy = accuracy_score(y_test, clf_entropy.predict(X_test))

# Put both models side by side in a small table - easy to read on a slide.
comparison = pd.DataFrame(
    {
        "Criterion": ["gini", "entropy"],
        "Test accuracy": [round(acc_gini, 4), round(acc_entropy, 4)],
        "Tree depth": [clf_gini.get_depth(), clf_entropy.get_depth()],
        "Leaves": [clf_gini.get_n_leaves(), clf_entropy.get_n_leaves()],
    }
)
print("\nGini vs Entropy on the same train/test split:")
print(comparison.to_string(index=False))
print("\nTakeaway: the two criteria usually agree. Pick gini (the default) unless")
print("you have a specific reason not to - the difference here is not meaningful.")


print("\n" + "=" * 78)
print("PART 3 - HYPERPARAMETERS AND OVERFITTING")
print("=" * 78)

# =============================================================================
# SECTION 3.1 - TRAINING vs TESTING ACCURACY ACROSS max_depth
# -----------------------------------------------------------------------------
# WHAT: grow trees of depth 1, 2, ... 10 and record BOTH accuracies.
# WHY : this is the clearest picture of overfitting you can show a class.
#       Training accuracy climbs towards 100% as the tree is allowed to grow,
#       because a deep tree can carve out a box around every single training
#       point. Testing accuracy stops improving - and may dip - once the tree
#       starts memorising noise instead of learning the pattern.
#       The best depth is where the test line flattens, NOT where train peaks.
# =============================================================================

depths = range(1, 11)                   # max_depth values to try
train_scores, test_scores = [], []      # collect accuracies here

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, criterion="gini", random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))   # .score() = accuracy
    test_scores.append(model.score(X_test, y_test))

# Print the numbers too - the chart is nicer, but the table is quotable.
print("\nAccuracy vs max_depth:")
print(pd.DataFrame({
    "max_depth": list(depths),
    "train_acc": np.round(train_scores, 4),
    "test_acc": np.round(test_scores, 4),
}).to_string(index=False))

plt.figure(figsize=(9, 5))
# Two series -> two colours AND two different markers/line styles, so the chart
# still works in black-and-white or for a colour-blind viewer.
plt.plot(depths, train_scores, marker="o", linewidth=2, markersize=8,
         color=COLOR_TRAIN, label="Training accuracy")
plt.plot(depths, test_scores, marker="s", linewidth=2, markersize=8,
         linestyle="--", color=COLOR_TEST, label="Testing accuracy")
plt.title("Overfitting Curve - Deeper Trees Memorise the Training Data")
plt.xlabel("max_depth (how many questions the tree may ask in a row)")
plt.ylabel("Accuracy")
plt.xticks(list(depths))
plt.ylim(0.5, 1.05)
plt.legend(loc="lower right")           # legend is mandatory with 2+ series
show("Train vs test accuracy by depth")

# Report where the test set was happiest (ties -> the SMALLEST depth wins,
# because a simpler model that scores the same is the better model).
best_depth = depths[int(np.argmax(test_scores))]
print(f"\nSmallest depth reaching the best test accuracy: max_depth = {best_depth}")

# =============================================================================
# SECTION 3.2 - A DELIBERATELY SMALL, READABLE TREE
# -----------------------------------------------------------------------------
# WHAT: restrict the tree with three classic hyperparameters.
#         max_depth=3        -> at most 3 questions from root to leaf
#         min_samples_split=10 -> never split a node holding fewer than 10 flowers
#         min_samples_leaf=5   -> every leaf must keep at least 5 flowers
# WHY : these are "pre-pruning" rules - they stop the tree growing in the first
#       place. The result is a model a human can actually read out loud, which
#       is often worth a percentage point of accuracy in the real world.
# =============================================================================

clf_small = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion="gini",
    random_state=42,
)
clf_small.fit(X_train, y_train)

print(f"\nSmall tree -> depth {clf_small.get_depth()}, "
      f"{clf_small.get_n_leaves()} leaves, "
      f"test accuracy {clf_small.score(X_test, y_test):.4f}")

plt.figure(figsize=(13, 7))
plot_tree(
    clf_small,
    feature_names=iris.feature_names,
    class_names=list(iris.target_names),
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Pre-pruned Tree (max_depth=3, min_samples_split=10, min_samples_leaf=5)")
show("Small pre-pruned tree")


print("\n" + "=" * 78)
print("PART 4 - COST COMPLEXITY PRUNING (post-pruning)")
print("=" * 78)

# =============================================================================
# SECTION 4.1 - THE PRUNING PATH
# -----------------------------------------------------------------------------
# WHAT: CART's own pruning method. It scores a tree as
#           cost = (total impurity of the leaves) + alpha * (number of leaves)
#       alpha is a price you charge per leaf. alpha = 0 -> keep the full tree.
#       As alpha rises, branches that do not "pay for themselves" are cut off.
# WHY : unlike max_depth (a guess), this grows the full tree first and then
#       removes the weakest branches - usually a better-shaped final model.
# cost_complexity_pruning_path() returns exactly the alpha values at which the
# tree actually changes, so we only test alphas that matter.
# =============================================================================

path = clf_gini.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]       # drop the last alpha: it collapses the
                                        # tree to a single root node (useless)
impurities = path.impurities[:-1]

print(f"\nNumber of candidate alpha values: {len(ccp_alphas)}")
print(f"Alpha range: {ccp_alphas.min():.5f} to {ccp_alphas.max():.5f}")

# =============================================================================
# SECTION 4.2 - TRAIN ONE TREE PER ALPHA AND COMPARE
# -----------------------------------------------------------------------------
# WHAT: same loop idea as the depth experiment, but the knob is ccp_alpha.
# WHY : it shows the same story from a different angle - a bigger penalty gives
#       a smaller tree; too big and the model becomes too simple (underfits).
# =============================================================================

pruned_trees = []
for alpha in ccp_alphas:
    t = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    t.fit(X_train, y_train)
    pruned_trees.append(t)

train_acc_alpha = [t.score(X_train, y_train) for t in pruned_trees]
test_acc_alpha = [t.score(X_test, y_test) for t in pruned_trees]
leaf_counts = [t.get_n_leaves() for t in pruned_trees]

print("\nPruning results:")
print(pd.DataFrame({
    "ccp_alpha": np.round(ccp_alphas, 5),
    "leaves": leaf_counts,
    "train_acc": np.round(train_acc_alpha, 4),
    "test_acc": np.round(test_acc_alpha, 4),
}).to_string(index=False))

plt.figure(figsize=(9, 5))
plt.plot(ccp_alphas, train_acc_alpha, marker="o", linewidth=2, markersize=8,
         color=COLOR_TRAIN, label="Training accuracy", drawstyle="steps-post")
plt.plot(ccp_alphas, test_acc_alpha, marker="s", linewidth=2, markersize=8,
         linestyle="--", color=COLOR_TEST, label="Testing accuracy",
         drawstyle="steps-post")
plt.title("Cost Complexity Pruning - Bigger alpha, Smaller Tree")
plt.xlabel("ccp_alpha (price charged per leaf)")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
show("Accuracy vs ccp_alpha")

# =============================================================================
# SECTION 4.3 - PICK THE BEST ALPHA AND SHOW THE PRUNED TREE
# -----------------------------------------------------------------------------
# WHAT: choose the alpha with the highest test accuracy; on a tie take the
#       LARGEST alpha, because that gives the simplest tree for the same score.
# WHY : "simplest model that performs as well" is Occam's razor, and it is the
#       standard rule for choosing among equally good trees.
# HONESTY NOTE: we are selecting a hyperparameter using the test set here, which
#       slightly flatters the final number. In a real project you would pick
#       alpha with cross-validation on the TRAINING data and keep the test set
#       untouched until the very end.
# =============================================================================

best_idx = int(np.max(np.flatnonzero(                   # largest index that is
    np.isclose(test_acc_alpha, np.max(test_acc_alpha))  # tied for the best score
)))
best_alpha = ccp_alphas[best_idx]
clf_pruned = pruned_trees[best_idx]

print(f"\nBest ccp_alpha        : {best_alpha:.5f}")
print(f"Pruned tree depth     : {clf_pruned.get_depth()}")
print(f"Pruned tree leaves    : {clf_pruned.get_n_leaves()}  "
      f"(full tree had {clf_gini.get_n_leaves()})")
print(f"Pruned test accuracy  : {clf_pruned.score(X_test, y_test):.4f}  "
      f"(full tree: {acc_gini:.4f})")

plt.figure(figsize=(13, 7))
plot_tree(
    clf_pruned,
    feature_names=iris.feature_names,
    class_names=list(iris.target_names),
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title(f"Post-pruned Tree (ccp_alpha = {best_alpha:.5f})")
show("Pruned tree")


print("\n" + "=" * 78)
print("PART 5 - CART FOR REGRESSION (Diabetes dataset)")
print("=" * 78)

# =============================================================================
# SECTION 5.1 - LOAD THE DIABETES DATA
# -----------------------------------------------------------------------------
# WHAT: 442 patients, 10 standardised medical measurements, and a NUMBER as the
#       target: disease progression one year after baseline (roughly 25-346).
# WHY : the target is continuous, not a category, so this is REGRESSION. The
#       tree works identically - only the split rule and the leaf value change.
#       A regression leaf predicts the MEAN of the training targets that landed
#       in it, which is why a regression tree produces a staircase, not a line.
# =============================================================================

diabetes = load_diabetes()
diab_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diab_df["target"] = diabetes.target

print("\nFirst 5 rows of the Diabetes dataset:")
print(diab_df.head())
print(f"\nShape: {diab_df.shape}")
print(f"Target range: {diabetes.target.min():.0f} to {diabetes.target.max():.0f}, "
      f"mean {diabetes.target.mean():.1f}")

# =============================================================================
# SECTION 5.2 - SPLIT AND TRAIN THE REGRESSION TREE
# -----------------------------------------------------------------------------
# WHAT: criterion="squared_error" means each split is chosen to minimise the
#       variance (spread) of the target inside the two child groups.
# WHY : max_depth=4 keeps the tree small enough to print on a slide AND stops
#       the classic regression-tree failure mode: with no depth limit every
#       leaf ends up holding a single patient, giving perfect training error
#       and terrible test error.
# =============================================================================

Xr, yr = diabetes.data, diabetes.target
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)

reg = DecisionTreeRegressor(
    criterion="squared_error",      # minimise variance in the child nodes
    max_depth=4,                    # keep it readable and avoid overfitting
    random_state=42,
)
reg.fit(Xr_train, yr_train)
yr_pred = reg.predict(Xr_test)      # predicted disease progression

# =============================================================================
# SECTION 5.3 - REGRESSION SCORES
# -----------------------------------------------------------------------------
# MSE  : average squared error. Units are "target squared", so hard to feel.
# RMSE : square root of MSE - back in the target's own units. "On average we
#        are off by about this much."
# R^2  : fraction of the variance explained. 1.0 = perfect, 0.0 = no better
#        than always predicting the mean, negative = worse than the mean.
# =============================================================================

mse = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)                      # works on every sklearn version
r2 = r2_score(yr_test, yr_pred)

print(f"\nMSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}   (typical error, in the target's own units)")
print(f"R^2  : {r2:.4f}   (1.0 would be perfect)")
print(f"\nFor comparison, always guessing the training mean "
      f"({yr_train.mean():.1f}) would give "
      f"RMSE {np.sqrt(mean_squared_error(yr_test, np.full_like(yr_test, yr_train.mean()))):.2f}")

# =============================================================================
# SECTION 5.4 - ACTUAL vs PREDICTED SCATTER
# -----------------------------------------------------------------------------
# WHAT: one dot per test patient. x = the truth, y = what the tree said.
# WHY : the dashed diagonal is "perfect prediction". Points above the line are
#       over-estimates, below are under-estimates. Because a regression tree
#       only has a few leaves, the dots form HORIZONTAL BANDS - every patient
#       reaching the same leaf gets the exact same predicted number. That
#       staircase behaviour is the key visual lesson of this chart.
# =============================================================================

plt.figure(figsize=(7, 6.5))
plt.scatter(
    yr_test, yr_pred,
    s=55, alpha=0.75,
    color=COLOR_TRAIN,
    edgecolor="white", linewidth=1.2,   # thin ring so overlapping dots separate
)
# The reference line spans the full range of both axes.
lo = min(yr_test.min(), yr_pred.min())
hi = max(yr_test.max(), yr_pred.max())
plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2,
         color=COLOR_TEST, label="Perfect prediction (y = x)")
plt.title(f"Regression Tree: Actual vs Predicted  (R^2 = {r2:.3f}, RMSE = {rmse:.1f})")
plt.xlabel("Actual disease progression")
plt.ylabel("Predicted disease progression")
plt.legend(loc="upper left")
show("Actual vs predicted scatter")

# =============================================================================
# SECTION 5.5 - DRAW THE REGRESSION TREE
# -----------------------------------------------------------------------------
# WHAT: same picture as before, but each box now shows
#         squared_error : the variance inside that node (the thing being reduced)
#         samples       : patients that reached it
#         value         : the MEAN target - and for a leaf, the prediction itself
# WHY : it makes "a regression tree predicts group averages" concrete.
# =============================================================================

plt.figure(figsize=(18, 9))
plot_tree(
    reg,
    feature_names=diabetes.feature_names,
    filled=True,                    # darker = higher predicted value
    rounded=True,
    fontsize=8,
    precision=1,                    # fewer decimals -> less clutter
)
plt.title("CART Regression Tree (max_depth=4) - Diabetes")
show("Regression tree")

# Which measurements drove the regression? (same idea as Section 1.9)
reg_importance = pd.Series(
    reg.feature_importances_, index=diabetes.feature_names
).sort_values(ascending=False)
print("\nTop regression feature importances:")
print(reg_importance.head(5))


# =============================================================================
# SECTION 6 - PLAIN-ENGLISH SUMMARY (read this part out in the presentation)
# =============================================================================

summary = f"""
================================================================================
 WHAT THE RESULTS SHOW - IN PLAIN ENGLISH
================================================================================

1. CART IS ONE ALGORITHM, TWO JOBS.
   The same "ask a yes/no question about one feature, then split" idea solved
   both problems. Only the scoring rule changed: Gini/Entropy for classifying
   species, squared error for predicting a number.

2. CLASSIFICATION WORKED VERY WELL (Iris).
   The Gini tree scored {acc_gini * 100:.1f}% on the 30 held-out flowers, using a tree of
   depth {clf_gini.get_depth()} with {clf_gini.get_n_leaves()} leaves. Feature importance showed that petal length
   and petal width do nearly all the work - the sepal measurements are close to
   useless here. Any confusion that appears is between versicolor and virginica,
   which genuinely overlap; setosa is separated perfectly by a single question.

3. GINI vs ENTROPY MADE ALMOST NO DIFFERENCE.
   Gini: {acc_gini:.4f} accuracy, depth {clf_gini.get_depth()}, {clf_gini.get_n_leaves()} leaves.
   Entropy: {acc_entropy:.4f} accuracy, depth {clf_entropy.get_depth()}, {clf_entropy.get_n_leaves()} leaves.
   They measure the same idea ("how mixed is this group?") in slightly different
   maths. Gini is the default because it avoids a logarithm and is faster. Do
   not expect to win accuracy by switching.

4. DEEPER IS NOT BETTER - THAT IS OVERFITTING.
   In the depth experiment the TRAINING accuracy kept climbing towards 100%
   while the TESTING accuracy flattened out around max_depth = {best_depth}. A deep tree
   is just memorising individual training flowers. The gap between the blue and
   the orange line IS overfitting, drawn to scale.

5. PRUNING GAVE A SMALLER MODEL FOR THE SAME QUALITY.
   Cost complexity pruning charged a price (alpha) per leaf and cut the branches
   that did not earn it. At alpha = {best_alpha:.5f} the tree shrank from
   {clf_gini.get_n_leaves()} leaves to {clf_pruned.get_n_leaves()} while test accuracy stayed at {clf_pruned.score(X_test, y_test):.4f}.
   A smaller tree that scores the same is the better tree: easier to explain,
   and less likely to be relying on noise.

6. REGRESSION WAS MUCH HARDER - AND THAT IS THE HONEST RESULT.
   On the Diabetes data the tree got R^2 = {r2:.3f} and RMSE = {rmse:.1f}. That means it
   explains only about {max(r2, 0) * 100:.0f}% of the variation in disease progression. The
   scatter plot shows horizontal bands because a regression tree can only ever
   predict one value per leaf - it approximates a smooth relationship with a
   staircase. For continuous targets a single tree is usually a weak model;
   this is exactly the weakness that Random Forests and Gradient Boosting fix
   by averaging many trees.

7. *** WHY YOU SHOULD NOT BRAG ABOUT HIGH IRIS ACCURACY ***
   The Iris test set contains only 30 flowers, about 10 per species. With 30
   samples, ONE extra mistake moves the score by 3.3 percentage points - so
   "{acc_gini * 100:.1f}%" and "93.3%" are not meaningfully different results, they are
   the same model on a slightly different day. If the score is 100%, it does
   NOT mean the model is perfect; it means the model made zero mistakes on
   thirty easy, clean, famous flowers. Change random_state=42 to another number
   and the score will move around.
   Three reasons to stay cautious:
     (a) Small test set  -> huge uncertainty on the estimate.
     (b) Iris is easy    -> the classes are almost linearly separable; real
                            data (medical, financial, text) is messy and
                            overlapping.
     (c) One split only  -> a single train/test split is one experiment. The
                            professional fix is k-fold cross-validation, which
                            reports a mean accuracy AND a standard deviation.

 THE ONE-SENTENCE CONCLUSION:
   CART is valuable because you can read the model, but it overfits by default,
   so the real skill is controlling tree size (max_depth / min_samples_* /
   ccp_alpha) and judging it on data it has never seen.
================================================================================
"""
print(summary)
