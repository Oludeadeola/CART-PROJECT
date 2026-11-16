# CART: Classification and Regression Trees — Interactive Demo

An interactive Streamlit app for a university Machine Learning presentation. Move a
slider, and the decision tree, the scores, the decision boundary and the overfitting
curve all redraw immediately.

| File | What it is |
|---|---|
| `app.py` | The Streamlit app — this is what you present. |
| `cart_demo.py` | A plain script version (no Streamlit) for Colab or a notebook. |
| `requirements.txt` | The six libraries the app needs. |

---

## What the app shows

1. **Metric cards** — accuracy (classification) or MSE / RMSE / R² (regression), plus
   tree depth and number of leaves.
2. **The tree diagram** — the actual rules the model learned, via `plot_tree(filled=True)`.
3. **A diagnostic chart** — confusion-matrix heatmap for classification, actual-vs-predicted
   scatter plus a residual histogram for regression.
4. **The overfitting curve** — training vs testing score for `max_depth` 1–10, with your
   currently selected depth marked by a dotted line.
5. **The decision boundary** (Iris only) — petal length × petal width, showing that a
   CART tree can only ever cut the space into axis-aligned **rectangles**.
6. **Feature importance** — a bar chart with the value printed on each bar.
7. **Try a prediction** — enter measurements and the app prints the exact chain of
   questions the flower answered on its way to a leaf.

---

## Run it locally

### 1. Install Python
Python 3.9 or newer. Check with:

```bash
python --version
```

> **On Windows**, if `python` opens the Microsoft Store instead of running, use `py`
> everywhere below in place of `python`.

### 2. Open a terminal in this folder

```bash
cd path/to/CART
```

### 3. Create a virtual environment (recommended, not required)

A virtual environment keeps these libraries out of your system Python.

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You will see `(.venv)` at the start of your prompt when it is active.

### 4. Install the libraries

```bash
pip install -r requirements.txt
```

This takes a minute or two the first time — scikit-learn and matplotlib are large.

### 5. Run the app

```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501** automatically. If it does not, copy that
address from the terminal into your browser.

Press **Ctrl+C** in the terminal to stop the app.

### Presenter tip
Edit `app.py` while the app is running and save — Streamlit notices and offers to rerun.
Also press **"⋮ → Settings → Wide mode"** and zoom your browser to ~80 % so the tree
diagram fills the projector screen.

---

## Deploy it free on Streamlit Community Cloud

This gives you a public URL you can put on a slide, so the class can open the app on
their own phones while you present. It is free and takes about five minutes.

### 1. Put the project on GitHub

Create a **public** repository on [github.com](https://github.com/new), then:

```bash
git init
git add app.py requirements.txt README.md cart_demo.py
git commit -m "CART interactive demo"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

Your repository must contain **`app.py` and `requirements.txt` at the top level**. That
is the only requirement — there is no config file to write.

### 2. Connect Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**.
2. Click **Sign in with GitHub** and authorise it.
3. Click **Create app** → **Deploy a public app from GitHub**.

### 3. Fill in the three fields

| Field | Value |
|---|---|
| Repository | `YOUR-USERNAME/YOUR-REPO` |
| Branch | `main` |
| Main file path | `app.py` |

Optionally click **Advanced settings** and set the Python version to 3.11.

### 4. Click Deploy

The build log appears on the right. It installs everything in `requirements.txt`, which
takes two to five minutes on the first deploy. When it finishes you get a URL like:

```
https://your-app-name.streamlit.app
```

That link is public and permanent. **Every `git push` to `main` redeploys it
automatically.**

### Things that commonly go wrong

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError` in the build log | A library is missing from `requirements.txt`. Add it, commit, push. |
| "Main file does not exist" | `app.py` is in a subfolder. Either move it to the repo root, or set *Main file path* to `folder/app.py`. |
| App shows "Zzz… this app has gone to sleep" | Free apps sleep after ~7 days idle. Click the wake button — it takes ~30 seconds. **Open and wake your app 10 minutes before the presentation.** |
| Blank page or a stale layout | Hard-refresh the browser (Ctrl+Shift+R). |
| Build is very slow | Normal on the first deploy; later deploys reuse a cache. |

---

## A note on the numbers you will see

The Iris test set is small — at a 20 % test size it holds **30 flowers, about 10 per
species**. One extra mistake moves the accuracy by 3.3 percentage points. So a 100 %
score does not mean the model is perfect; it means it made zero mistakes on thirty easy,
clean, famous flowers. Drag the **test size** slider and watch the accuracy jump around:
that instability *is* the lesson. In real work you would report a k-fold
cross-validation mean and standard deviation instead of a single number.
