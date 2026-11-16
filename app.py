"""
================================================================================
 CART: Classification and Regression Trees - INTERACTIVE STREAMLIT DEMO
================================================================================
 Run locally:   streamlit run app.py
 See README.md for setup and free deployment instructions.

 WHAT THIS APP IS FOR
   A decision tree is one of the few machine-learning models you can actually
   LOOK AT. This app lets the audience move a slider and watch the tree, the
   accuracy and the decision boundary change in real time - which makes the
   ideas of "underfitting", "overfitting" and "pruning" visible instead of
   abstract.

 HOW STREAMLIT WORKS (worth saying out loud in the presentation)
   Streamlit re-runs this entire file from top to bottom every time any widget
   changes. There is no callback code and no "update" button - the script IS
   the app. That is why the page refreshes automatically.
================================================================================
"""

# =============================================================================
# SECTION 0 - IMPORTS, PAGE CONFIG AND SHARED STYLE
# -----------------------------------------------------------------------------
# WHAT: load libraries once, set the browser tab title and the wide layout.
# WHY : st.set_page_config() must be the FIRST Streamlit command in the script,
#       otherwise Streamlit raises an error.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw to memory, not to a window:
                                           # required on a headless cloud server
import matplotlib.pyplot as plt
# Imported explicitly so we can guarantee the figure has an Agg canvas.
# scikit-learn's plot_tree needs canvas.get_renderer() to measure the text it
# draws, and the plain base canvas does not provide it.
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
import seaborn as sns
import streamlit as st

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_text,     # plain-text version of a tree, used as a safe fallback
    plot_tree,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)

st.set_page_config(
    page_title="CART Interactive Demo",
    page_icon="🌳",
    layout="wide",                         # use the full browser width
)

# --- Hide the hosting platform's own chrome ---------------------------------
# Streamlit Community Cloud adds a toolbar and badges around the app. Those
# include a link back to the source repository, which also shows the account
# name. This CSS removes them so viewers see only the demo itself.
# NOTE: the owner still sees the developer tools while signed in to Streamlit.
# To check what the audience actually sees, open the app in a private window.
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden; height: 0;}
      header [data-testid="stToolbar"] {visibility: hidden; height: 0;}
      [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
      [data-testid="stDecoration"] {display: none;}
      [data-testid="stStatusWidget"] {visibility: hidden; height: 0;}
      [data-testid="manage-app-button"] {display: none;}
      .viewerBadge_container__1QSob, .viewerBadge_link__qRIco {display: none;}
      footer {visibility: hidden; height: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- A small, fixed, colour-blind-safe palette -------------------------------
# One colour per class, always in the same order, so a colour never changes
# meaning while the audience is watching. Blue = training, orange = testing.
CLASS_COLORS = ["#2A6FB0", "#D9822B", "#2E9E8F"]   # setosa / versicolor / virginica
CLASS_MARKERS = ["o", "s", "^"]                    # shape backs up the colour,
                                                   # so the chart still works in
                                                   # greyscale or for a colour-
                                                   # blind viewer
COLOR_TRAIN = "#2A6FB0"
COLOR_TEST = "#D9822B"
INK = "#333333"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["font.size"] = 9


def render(fig):
    """Draw a matplotlib figure in the page, then free its memory.

    WHY: Streamlit re-runs the script on every interaction. Without the
    plt.close() the app would slowly leak figures and warn about it."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# SECTION 1 - LOADING THE DATA (CACHED)
# -----------------------------------------------------------------------------
# WHAT: load Iris (classification) or Diabetes (regression) from scikit-learn.
# WHY : @st.cache_data stores the result the first time and reuses it on every
#       later re-run. Without it we would reload the dataset on every single
#       slider move. (It is cheap here, but it is the correct habit.)
# =============================================================================

@st.cache_data
def load_classification_data():
    """Iris: 150 flowers, 4 measurements, 3 species."""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, list(data.feature_names), list(data.target_names)


@st.cache_data
def load_regression_data():
    """Diabetes: 442 patients, 10 standardised measurements, a NUMBER target
    (disease progression one year after baseline)."""
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, list(data.feature_names)


# =============================================================================
# SECTION 2 - SIDEBAR: ALL THE CONTROLS
# -----------------------------------------------------------------------------
# WHAT: every knob the audience can turn lives here.
# WHY : keeping controls on the left and results on the right is the standard
#       dashboard layout - the viewer always knows where the cause and the
#       effect are.
# =============================================================================

st.sidebar.title("⚙️ Controls")
st.sidebar.caption("Change anything - the page updates immediately.")

# --- Mode -------------------------------------------------------------------
mode = st.sidebar.radio(
    "Mode",
    ["Classification (Iris)", "Regression (Diabetes)"],
    help="Classification predicts a CATEGORY (which species). "
         "Regression predicts a NUMBER.",
)
is_classification = mode.startswith("Classification")

# --- Criterion --------------------------------------------------------------
# The criterion is the rule CART uses to score a candidate split.
if is_classification:
    criterion = st.sidebar.selectbox(
        "Splitting criterion",
        ["gini", "entropy"],
        help="Gini = 1 - Σp².  Entropy = -Σp·log₂(p).  "
             "Both measure how 'mixed' a node is; they almost always agree.",
    )
else:
    # Regression trees minimise variance, so there is only one sensible choice
    # for this demo. The disabled widget keeps the sidebar layout stable.
    criterion = "squared_error"
    st.sidebar.selectbox(
        "Splitting criterion",
        ["squared_error"],
        disabled=True,
        help="Regression splits are chosen to reduce the squared error "
             "(the spread of the target) inside the two child nodes.",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Tree size (pre-pruning)")

# --- The three classic pre-pruning hyperparameters ---------------------------
max_depth = st.sidebar.slider(
    "max_depth", 1, 10, 3,
    help="How many questions the tree may ask in a row. "
         "Bigger = more complex = more likely to overfit.",
)
min_samples_split = st.sidebar.slider(
    "min_samples_split", 2, 20, 2,
    help="A node must hold at least this many samples before it is allowed "
         "to split at all.",
)
min_samples_leaf = st.sidebar.slider(
    "min_samples_leaf", 1, 20, 1,
    help="Every leaf must keep at least this many samples. "
         "Raising it is a strong, simple defence against overfitting.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Pruning & data split")

ccp_alpha = st.sidebar.slider(
    "ccp_alpha (cost complexity pruning)", 0.0, 0.05, 0.0, step=0.001,
    format="%.3f",
    help="A price charged per leaf. The tree is grown fully and then branches "
         "that do not earn their price are cut off. 0.0 = no pruning.",
)
test_size = st.sidebar.slider(
    "Test set size", 0.10, 0.40, 0.20, step=0.05, format="%.0f%%",
    help="The share of the data hidden from the model and used only for "
         "marking. A smaller test set gives a noisier, less trustworthy score.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "random_state is fixed at 42 everywhere, so every run gives identical "
    "numbers - important when the class follows along on their own laptops."
)

RANDOM_STATE = 42


# =============================================================================
# SECTION 3 - PREPARE DATA AND TRAIN THE MODEL
# -----------------------------------------------------------------------------
# WHAT: split the data, then fit a tree using the sidebar settings.
# WHY : this block runs on EVERY interaction. The whole app is really just
#       "read the sliders -> fit a tree -> draw it", repeated.
# =============================================================================

if is_classification:
    df, feature_names, class_names = load_classification_data()
    X = df[feature_names].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,          # keep the 3 species balanced in both halves
    )

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=RANDOM_STATE,
    )
else:
    df, feature_names = load_regression_data()
    class_names = None
    X = df[feature_names].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    model = DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=RANDOM_STATE,
    )

model.fit(X_train, y_train)          # <- this one line is the CART algorithm
y_pred = model.predict(X_test)


# =============================================================================
# SECTION 4 - HEADER AND METRIC CARDS
# -----------------------------------------------------------------------------
# WHAT: the headline numbers, big and at the top.
# WHY : a dashboard should answer "how good is it?" before it shows any chart.
#       st.metric() renders the large-number "card" style automatically.
# =============================================================================

st.title("CART: Classification and Regression Trees – Interactive Demo")
st.caption(
    "One algorithm, two jobs. Move the sliders on the left and watch the tree, "
    "the score and the decision boundary change."
)

if is_classification:
    acc_test = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, model.predict(X_train))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{acc_test:.1%}",
              delta=f"{acc_test - acc_train:+.1%} vs train",
              # A NEGATIVE gap means the test score sits below the training
              # score - that is overfitting, and it must look BAD (red).
              # "normal" = negative shows red. Do NOT use "inverse" here.
              delta_color="normal",
              help="Share of unseen flowers classified correctly. The delta "
                   "shows the gap to training accuracy; a big negative gap "
                   "means the tree is memorising.")
    c2.metric("Train accuracy", f"{acc_train:.1%}")
    c3.metric("Tree depth", model.get_depth(),
              help="Longest chain of questions from the root to a leaf.")
    c4.metric("Leaves", model.get_n_leaves(),
              help="Number of final decision boxes = how many different "
                   "answers the tree can give.")
else:
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MSE", f"{mse:,.1f}",
              help="Mean squared error - average of (actual − predicted)². "
                   "Lower is better.")
    c2.metric("RMSE", f"{rmse:,.1f}",
              help="Square root of MSE, back in the target's own units: "
                   "'on average we are off by about this much'.")
    c3.metric("R²", f"{r2:.3f}",
              help="Share of the variation explained. 1.0 = perfect, "
                   "0.0 = no better than always guessing the mean.")
    c4.metric("Tree depth", model.get_depth())
    c5.metric("Leaves", model.get_n_leaves())

st.markdown(
    f"**Dataset:** {len(X)} rows, {len(feature_names)} features &nbsp;•&nbsp; "
    f"**Training:** {len(X_train)} rows &nbsp;•&nbsp; "
    f"**Testing:** {len(X_test)} rows &nbsp;•&nbsp; "
    f"**Criterion:** `{criterion}`"
)
st.divider()


# =============================================================================
# SECTION 5 - THE TREE DIAGRAM
# -----------------------------------------------------------------------------
# WHAT: the actual rules the model learned, drawn as a picture.
# WHY : this is the whole reason CART is still taught and still used. Unlike a
#       neural network, the model IS the explanation. Each box shows the
#       question, the impurity, how many samples arrived, and the prediction.
# =============================================================================

st.subheader("1. The decision tree the model learned")

# If the sliders have pruned the tree away to nothing, say so in words - an
# empty-looking diagram just confuses the audience.
if model.get_n_leaves() <= 1:
    st.warning(
        "**The tree has been pruned down to a single box.** With these settings "
        "it cannot ask any question at all, so it gives every row the same "
        "answer. That is *underfitting* — the opposite problem to overfitting. "
        "Lower `ccp_alpha` or `min_samples_leaf` to let it grow again."
    )
elif model.get_depth() >= 6:
    st.warning(
        "**This tree is now too big to read on a screen.** That is itself the "
        "lesson: once a tree needs this many questions, you have lost the one "
        "advantage trees have over other models — being readable by a human."
    )

def draw_tree_diagram(fitted_model, names, classes, subtitle):
    """Draw the tree picture, and never let a drawing problem kill the page.

    WHY THIS IS NOT JUST plot_tree():
      scikit-learn measures the text it is about to draw by asking the figure
      for a "renderer". That only exists on an Agg canvas. Depending on the
      matplotlib version and the server, a bare figure can come back with the
      base canvas instead, and plot_tree then fails.
      So we attach an Agg canvas ourselves, and if anything still goes wrong we
      fall back to a TEXT version of exactly the same tree. The audience still
      sees the rules; only the picture is lost.
    """
    fig, ax = plt.subplots(figsize=(16, max(5, min(9, fitted_model.get_depth() * 1.8))))
    FigureCanvasAgg(fig)                     # guarantees canvas.get_renderer()
    if not hasattr(fig.canvas, "get_renderer"):
        # Last-resort shim for matplotlib builds that renamed the method.
        fig.canvas.get_renderer = fig._get_renderer

    plot_tree(
        fitted_model,
        feature_names=names,
        class_names=classes,        # None for regression - sklearn handles that
        filled=True,                # colour each box by its majority class /
                                    # predicted value; darker = purer / higher
        rounded=True,
        fontsize=8,
        precision=2,
        ax=ax,
    )
    ax.set_title(subtitle)
    render(fig)


tree_subtitle = (
    f"CART tree — criterion={criterion}, max_depth={max_depth}, "
    f"min_samples_leaf={min_samples_leaf}, ccp_alpha={ccp_alpha:.3f}"
)

try:
    draw_tree_diagram(model, feature_names, class_names, tree_subtitle)
except Exception as exc:                     # noqa: BLE001 - we want every error
    plt.close("all")                         # drop the half-built figure
    st.warning(
        "The tree picture could not be drawn on this server, so here are the "
        "same rules written out as text. Every line is one question; the "
        "indentation shows the branches."
    )
    st.code(
        export_text(
            model,
            feature_names=list(feature_names),
            decimals=2,
            show_weights=True,
        ),
        language="text",
    )
    st.caption(f"(Technical detail, for debugging: {type(exc).__name__}: {exc})")

st.info(
    "**How to read a box.** The top line is the question — if it is true you go "
    "**left**, if false you go **right**. `samples` is how many training rows "
    "reached that box. For classification, `value` is the class counts and "
    "`class` is the majority (what a leaf predicts). For regression, `value` is "
    "the **average target** of the rows in that box — which is literally the "
    "prediction. Try setting `max_depth` to 1, then 2, then 10 and watch the "
    "tree grow."
)
st.divider()


# =============================================================================
# SECTION 6 - CONFUSION MATRIX  /  ACTUAL vs PREDICTED
# -----------------------------------------------------------------------------
# WHAT: the diagnostic chart appropriate to the task.
# WHY : a single score hides WHERE the model goes wrong. These two charts show
#       the shape of the errors, not just their size.
# =============================================================================

if is_classification:
    st.subheader("2. Where does it get things wrong?")
    left, right = st.columns([1, 1])

    with left:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm,
            annot=True, fmt="d",
            cmap="Blues",                  # one hue, light→dark = few→many
            cbar_kws={"label": "Number of flowers"},
            linewidths=2, linecolor="white",   # gap keeps the cells distinct
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title("Confusion Matrix (test set)")
        ax.set_xlabel("Predicted species")
        ax.set_ylabel("True species")
        render(fig)
        st.caption(
            "**In plain English:** rows are the truth, columns are the guess. "
            "The diagonal is correct answers; anything off it is a mistake, and "
            "it tells you exactly which two species got confused. Setosa is "
            "usually perfect — it is separable by one question. Versicolor and "
            "virginica genuinely overlap, so that is where errors appear."
        )

    with right:
        report = classification_report(
            y_test, y_pred, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        st.markdown("**Per-species report (test set)**")
        st.dataframe(
            pd.DataFrame(report).transpose().round(3),
        )
        st.caption(
            "**precision** = of the flowers I *called* this species, how many "
            "really were? **recall** = of the flowers that really *are* this "
            "species, how many did I catch? **support** = how many test flowers "
            "of that species exist — note how small that number is."
        )
else:
    st.subheader("2. Actual vs predicted")
    left, right = st.columns([3, 2])

    with left:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(
            y_test, y_pred,
            s=50, alpha=0.75, color=COLOR_TRAIN,
            edgecolor="white", linewidth=1.1,   # thin ring separates overlaps
            label="Test patients",
        )
        lo = float(min(y_test.min(), y_pred.min()))
        hi = float(max(y_test.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2,
                color=COLOR_TEST, label="Perfect prediction (y = x)")
        ax.set_title(f"Actual vs Predicted  (R² = {r2:.3f}, RMSE = {rmse:.1f})")
        ax.set_xlabel("Actual disease progression")
        ax.set_ylabel("Predicted disease progression")
        ax.legend(loc="upper left")
        render(fig)
        st.caption(
            "**In plain English:** each dot is one patient. The dashed line is a "
            "perfect prediction — dots above it are over-estimates, below are "
            "under-estimates. Notice the **horizontal bands**: a regression tree "
            "has only a few leaves, and every patient landing in the same leaf "
            "gets the *exact same* number. A tree approximates a smooth "
            "relationship with a staircase."
        )

    with right:
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(5, 5.5))
        ax.hist(residuals, bins=20, color=COLOR_TRAIN,
                edgecolor="white", linewidth=1)
        ax.axvline(0, color=COLOR_TEST, linestyle="--", linewidth=2)
        ax.set_title("Distribution of errors (residuals)")
        ax.set_xlabel("Actual − Predicted")
        ax.set_ylabel("Number of patients")
        render(fig)
        st.caption(
            "**In plain English:** errors should cluster around zero and look "
            "roughly symmetric. A long tail on one side means the model is "
            "systematically wrong for some group of patients."
        )

st.divider()


# =============================================================================
# SECTION 7 - THE OVERFITTING CURVE
# -----------------------------------------------------------------------------
# WHAT: refit the tree at every depth from 1 to 10 and plot BOTH scores.
# WHY : this is the clearest picture of overfitting available. Training score
#       climbs towards perfect because a deep tree can carve a box around every
#       single training row. Testing score flattens or falls, because those
#       extra boxes are fitted to noise, not to the pattern.
# NOTE: @st.cache_data means this 10-model loop only re-runs when the data or
#       the other hyperparameters actually change - not on every mouse move.
# =============================================================================

st.subheader("3. Why deeper is not better — the overfitting curve")


@st.cache_data(show_spinner=False)
def depth_curve(X_tr, X_te, y_tr, y_te, is_clf, crit, mss, msl, alpha):
    """Fit one tree per depth and return train/test scores.

    The arrays are passed in as arguments (not read from globals) so that
    Streamlit's cache can tell when the inputs have genuinely changed."""
    depths = list(range(1, 11))
    tr, te = [], []
    for d in depths:
        Model = DecisionTreeClassifier if is_clf else DecisionTreeRegressor
        m = Model(
            criterion=crit, max_depth=d,
            min_samples_split=mss, min_samples_leaf=msl,
            ccp_alpha=alpha, random_state=RANDOM_STATE,
        )
        m.fit(X_tr, y_tr)
        tr.append(m.score(X_tr, y_tr))     # accuracy for a classifier, R² for a regressor
        te.append(m.score(X_te, y_te))
    return depths, tr, te


depths, train_scores, test_scores = depth_curve(
    X_train, X_test, y_train, y_test,
    is_classification, criterion, min_samples_split, min_samples_leaf, ccp_alpha,
)
score_label = "Accuracy" if is_classification else "R² score"

curve_left, curve_right = st.columns([3, 2])

with curve_left:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    # Two series: different colour AND different marker AND different line
    # style, so the chart survives a projector, greyscale or colour blindness.
    ax.plot(depths, train_scores, marker="o", markersize=7, linewidth=2,
            color=COLOR_TRAIN, label="Training score")
    ax.plot(depths, test_scores, marker="s", markersize=7, linewidth=2,
            linestyle="--", color=COLOR_TEST, label="Testing score")
    # Highlight the depth currently selected in the sidebar.
    ax.axvline(max_depth, color=INK, linestyle=":", linewidth=2, alpha=0.8)
    ax.annotate(
        f"you are here\nmax_depth = {max_depth}",
        xy=(max_depth, min(test_scores)),
        xytext=(5, 8), textcoords="offset points",
        fontsize=9, color=INK, fontweight="bold",
    )
    ax.set_title(f"Training vs Testing {score_label} by max_depth")
    ax.set_xlabel("max_depth (how many questions in a row the tree may ask)")
    ax.set_ylabel(score_label)
    ax.set_xticks(depths)
    ax.legend(loc="lower right")           # 2 series → a legend is mandatory
    render(fig)

with curve_right:
    best_idx = int(np.argmax(test_scores))
    st.markdown(
        f"""
**In plain English.** The **blue** line is how well the tree does on data it has
already seen; the **orange** line is how well it does on data it has never seen.
Only the orange line is a real result.

- The blue line keeps climbing — a deeper tree can always memorise more.
- The orange line flattens out around **max_depth = {depths[best_idx]}** and then
  stops improving.
- **The gap between the two lines is overfitting, drawn to scale.**

The dotted vertical line marks the depth you have selected. Drag `max_depth` to
10 and watch the gap open up.
        """
    )
    st.dataframe(
        pd.DataFrame({
            "max_depth": depths,
            "train": np.round(train_scores, 3),
            "test": np.round(test_scores, 3),
        }).set_index("max_depth"),
        height=240,
    )

st.divider()


# =============================================================================
# SECTION 8 - 2D DECISION BOUNDARY (IRIS ONLY)
# -----------------------------------------------------------------------------
# WHAT: train a second tree on just petal length and petal width, then colour
#       every point of the plane by what that tree would predict there.
# WHY : it makes the defining property of CART visible - because every question
#       is "feature <= number", the boundaries are always straight lines
#       parallel to the axes, so the space is cut into RECTANGLES. A tree can
#       never draw a diagonal boundary; it can only approximate one with steps.
# NOTE: this is a SEPARATE 2-feature model. You cannot draw the 4-feature tree
#       on a flat screen, and pretending otherwise would be dishonest.
# =============================================================================

if is_classification:
    st.subheader("4. How the tree carves up the space (petal length × petal width)")

    # Indexes 2 and 3 of the Iris feature list are petal length and petal width.
    PETAL = [2, 3]
    X2_train, X2_test = X_train[:, PETAL], X_test[:, PETAL]

    model_2d = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=RANDOM_STATE,
    )
    model_2d.fit(X2_train, y_train)

    # Build a fine grid covering the whole plot area, then ask the tree to
    # classify every single grid point. That painted grid IS the boundary.
    pad = 0.4
    x_min, x_max = X[:, PETAL[0]].min() - pad, X[:, PETAL[0]].max() + pad
    y_min, y_max = X[:, PETAL[1]].min() - pad, X[:, PETAL[1]].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    bd_left, bd_right = st.columns([3, 2])

    with bd_left:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        # Pale filled regions in the background...
        ax.contourf(xx, yy, Z, alpha=0.18,
                    cmap=ListedColormap(CLASS_COLORS), levels=[-0.5, 0.5, 1.5, 2.5])
        # ...crisp lines on the boundaries themselves, so the rectangles read.
        ax.contour(xx, yy, Z, colors="white", linewidths=1.6,
                   levels=[0.5, 1.5])

        # The actual flowers on top. Training = filled, testing = white edge.
        for k, name in enumerate(class_names):
            m_tr = y_train == k
            ax.scatter(X2_train[m_tr, 0], X2_train[m_tr, 1],
                       c=CLASS_COLORS[k], marker=CLASS_MARKERS[k], s=45,
                       edgecolor="white", linewidth=1.0, label=f"{name} (train)")
            m_te = y_test == k
            ax.scatter(X2_test[m_te, 0], X2_test[m_te, 1],
                       facecolor="none", edgecolor=CLASS_COLORS[k],
                       marker=CLASS_MARKERS[k], s=80, linewidth=2,
                       label=f"{name} (test)")

        ax.set_title(f"Decision regions of a 2-feature tree "
                     f"(depth {model_2d.get_depth()}, {model_2d.get_n_leaves()} leaves)")
        ax.set_xlabel("Petal length (cm)")
        ax.set_ylabel("Petal width (cm)")
        ax.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)
        render(fig)

    with bd_right:
        st.markdown(
            """
**In plain English.** The shaded colour shows what the tree would predict at
every point of this plane. Solid markers are training flowers; hollow markers
are test flowers.

**The key observation for the presentation:** every boundary is a **straight
line parallel to an axis**. That is not a coincidence — it is forced by the
algorithm. Every question a CART tree can ask has the form
*"is this feature ≤ some number?"*, which is exactly a horizontal or vertical
cut. So the model can only ever produce **rectangles**.

- Low `max_depth` → a few big rectangles → it may underfit.
- High `max_depth` → thin slivers wrapped around individual points → that is
  overfitting, and you can literally see it.
- Raising `min_samples_leaf` or `ccp_alpha` forces the rectangles back together.

A diagonal relationship therefore has to be approximated with a staircase,
which is a real limitation of trees compared with, say, logistic regression.
            """
        )
    st.caption(
        "Note: this panel trains a **separate** tree on only these two features. "
        "The 4-feature tree shown above lives in 4 dimensions and cannot be "
        "drawn on a flat screen, so its boundary would not match this picture."
    )
    st.divider()


# =============================================================================
# SECTION 9 - FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
# WHAT: how much total impurity (or variance) each feature removed, scaled to
#       sum to 1.0 across all features.
# WHY : it answers "which measurement actually decides the answer?" This is a
#       free form of feature selection that trees hand you for nothing.
# CAUTION worth saying out loud: a feature can score 0.0 simply because a
#       correlated feature got picked first - zero importance does NOT prove
#       the feature is useless.
# =============================================================================

section_number = 5 if is_classification else 4
st.subheader(f"{section_number}. Which features is the tree actually using?")

importances = pd.Series(model.feature_importances_, index=feature_names)
importances = importances.sort_values()

imp_left, imp_right = st.columns([3, 2])

with imp_left:
    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.45 * len(feature_names) + 1.4)))
    bars = ax.barh(importances.index, importances.values,
                   color=COLOR_TRAIN, height=0.62)   # one series → no legend
    # Edge case a student WILL hit: if ccp_alpha is dragged high enough the
    # tree collapses to a single root node, every importance is 0.0, and the
    # chart would otherwise render with a nonsense axis.
    top = max(float(importances.values.max()), 1e-9)
    # Direct value labels: the reader should not have to measure against an axis.
    for bar, value in zip(bars, importances.values):
        ax.text(value + top * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, top * 1.2)
    ax.set_title("Feature importance (share of total impurity reduction)")
    ax.set_xlabel("Importance — all features sum to 1.0")
    ax.set_ylabel("Feature")
    render(fig)

with imp_right:
    st.markdown(
        """
**In plain English.** Every time the tree splits on a feature it removes some
"mess". Importance adds up all that cleanup per feature and scales it to 1.0.

For Iris you will find the **petal** measurements do nearly all the work and
sepal width is close to useless — which is a real biological result, not an
artefact.

⚠️ **One caution to mention:** a score of 0.00 does *not* prove a feature is
worthless. If two features carry the same information, the tree picks one and
the other gets nothing left to contribute.
        """
    )
    st.dataframe(
        importances.sort_values(ascending=False).round(4).rename("importance"),
    )

st.divider()


# =============================================================================
# SECTION 10 - "TRY A PREDICTION": FOLLOW ONE SAMPLE THROUGH THE TREE
# -----------------------------------------------------------------------------
# WHAT: the user enters measurements; we predict, and then print the exact
#       chain of questions the sample answered on its way to a leaf.
# WHY : this is the pay-off of the whole lecture. A tree can justify a single
#       prediction as a short list of if-statements - something a neural
#       network cannot do. decision_path() gives us the nodes visited.
# =============================================================================

section_number += 1
st.subheader(f"{section_number}. Try a prediction — and see the path the tree took")


def explain_path(fitted_model, sample_row, names, class_labels=None):
    """Return the list of questions a single sample answered, in order.

    HOW IT WORKS
      decision_path() returns a sparse matrix marking every node the sample
      visited. tree_.feature and tree_.threshold tell us the question stored at
      each internal node. A leaf is marked with feature = -2, so we stop there.
    """
    sample_row = np.asarray(sample_row, dtype=float).reshape(1, -1)
    indicator = fitted_model.decision_path(sample_row)
    # The indices of the visited nodes, root first, leaf last:
    node_ids = indicator.indices[indicator.indptr[0]:indicator.indptr[1]]

    tree = fitted_model.tree_
    steps = []
    for node in node_ids:
        if tree.feature[node] == -2:              # -2 = this node is a leaf
            n_samples = int(tree.n_node_samples[node])
            if class_labels is not None:
                raw = np.asarray(tree.value[node][0], dtype=float)
                # scikit-learn >= 1.4 stores class *proportions* here, while
                # older versions stored raw counts. Normalise so this works on
                # either version (Colab and Streamlit Cloud may differ).
                counts = raw * n_samples if raw.sum() <= 1.000001 else raw
                winner = class_labels[int(np.argmax(counts))]
                detail = (f"**LEAF** → predict **{winner}**  "
                          f"({n_samples} training flowers landed here; "
                          f"counts {np.rint(counts).astype(int).tolist()})")
            else:
                detail = (f"**LEAF** → predict **{tree.value[node][0][0]:.1f}**  "
                          f"(the average of the {n_samples} training rows here)")
            steps.append(detail)
            break

        feat_idx = int(tree.feature[node])
        thresh = tree.threshold[node]
        value = sample_row[0, feat_idx]
        went_left = value <= thresh
        steps.append(
            f"`{names[feat_idx]}` = **{value:.2f}** — is it ≤ **{thresh:.2f}**? "
            f"**{'YES → go LEFT' if went_left else 'NO → go RIGHT'}**"
        )
    return steps


if is_classification:
    st.markdown("Enter a flower's measurements and the tree will classify it.")
    input_cols = st.columns(4)
    sample = []
    for i, fname in enumerate(feature_names):
        with input_cols[i]:
            # Default to the dataset mean, bounded by the real observed range,
            # so a student cannot enter a nonsense flower by accident.
            sample.append(
                st.slider(
                    fname,
                    float(X[:, i].min()), float(X[:, i].max()),
                    float(np.round(X[:, i].mean(), 1)),
                    step=0.1,
                )
            )

    sample_arr = np.array(sample).reshape(1, -1)
    pred_idx = int(model.predict(sample_arr)[0])
    proba = model.predict_proba(sample_arr)[0]

    res_left, res_right = st.columns([2, 3])

    with res_left:
        st.success(f"### Predicted species: **{class_names[pred_idx]}**")
        st.markdown("**Confidence from the leaf:**")
        for k, name in enumerate(class_names):
            st.progress(float(proba[k]), text=f"{name} — {proba[k]:.0%}")
        st.caption(
            "These 'probabilities' are just the class proportions among the "
            "training flowers sitting in that leaf — not a calibrated "
            "probability. With a deep tree the leaf often holds one flower, so "
            "it reports a confident 100% on almost no evidence."
        )

    with res_right:
        st.markdown("**The path this flower took through the tree:**")
        for step_no, text in enumerate(
            explain_path(model, sample_arr, feature_names, class_names), start=1
        ):
            st.markdown(f"{step_no}. {text}")
        st.caption(
            "This short list of if-statements is the *entire* justification for "
            "the prediction. That is what people mean when they call decision "
            "trees an **interpretable** model — and it is why they are still "
            "used in medicine, credit scoring and anywhere a decision has to be "
            "defended."
        )
else:
    st.markdown(
        "Pick a patient from the test set and follow them through the tree. "
        "(The Diabetes features are pre-standardised, so raw sliders would be "
        "meaningless — using a real patient keeps the numbers honest.)"
    )
    patient_idx = st.number_input(
        "Test patient number", min_value=0, max_value=len(X_test) - 1,
        value=0, step=1,
    )
    sample_arr = X_test[int(patient_idx)].reshape(1, -1)
    prediction = float(model.predict(sample_arr)[0])
    actual = float(y_test[int(patient_idx)])

    res_left, res_right = st.columns([2, 3])

    with res_left:
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted", f"{prediction:.1f}")
        m2.metric("Actual", f"{actual:.1f}")
        m3.metric("Error", f"{actual - prediction:+.1f}")
        st.markdown("**This patient's measurements:**")
        st.dataframe(
            pd.Series(sample_arr[0], index=feature_names).round(4)
              .rename("standardised value"),
        )

    with res_right:
        st.markdown("**The path this patient took through the tree:**")
        for step_no, text in enumerate(
            explain_path(model, sample_arr, feature_names, None), start=1
        ):
            st.markdown(f"{step_no}. {text}")
        st.caption(
            "Notice the final step: the prediction is simply the **average** of "
            "the training patients who ended up in the same leaf. Every patient "
            "reaching that leaf gets the identical number — which is why the "
            "scatter plot above shows horizontal bands."
        )

st.divider()


# =============================================================================
# SECTION 11 - CLOSING SUMMARY
# -----------------------------------------------------------------------------
# WHAT: the take-home message, written so it can be read out loud.
# =============================================================================

with st.expander("📌 Summary — the five things to remember about CART", expanded=False):
    st.markdown(
        f"""
1. **One algorithm, two jobs.** The same "ask a yes/no question about one
   feature, then split" idea handles both tasks. Only the scoring rule changes:
   Gini or entropy for categories, squared error for numbers.

2. **It is readable, and that is the point.** You can print the model, read it
   aloud, and justify any single prediction as a short list of if-statements.
   Very few models can do that.

3. **It overfits by default.** Left unrestricted, a tree grows until every leaf
   is pure — which means it has memorised the training set. The overfitting
   curve above shows the training score and the testing score pulling apart.

4. **Controlling size is the real skill.** `max_depth`, `min_samples_split` and
   `min_samples_leaf` stop the tree growing (*pre-pruning*); `ccp_alpha` grows
   it fully and then cuts weak branches off (*post-pruning*, which is CART's
   own method). A smaller tree with the same score is the better tree.

5. **Be sceptical of a high score on a small test set.** Right now the test set
   holds **{len(X_test)} rows**. With a test set that small, one extra mistake
   moves the score by several percentage points — so 100% does not mean
   "perfect model", it means "zero mistakes on a handful of easy, clean, famous
   rows". Iris in particular is almost separable by a single measurement. The
   professional fix is **k-fold cross-validation**, which reports a mean score
   *and* a standard deviation instead of one lucky number.

> **One-sentence conclusion:** CART is valuable because you can read it, but it
> overfits by default — so the job is to control the tree's size and judge it
> only on data it has never seen.
        """
    )

st.caption(
    "Built with scikit-learn, pandas, numpy, matplotlib, seaborn and Streamlit. "
    "All randomness fixed with random_state=42."
)
