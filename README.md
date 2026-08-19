# 🧺 Basket DNA: Uncovering Hidden Purchase Patterns using Unsupervised Learning

Market Basket Analysis on the **Online Retail Dataset** using the **Apriori Algorithm**,
plus a live **Streamlit recommendation dashboard** built on top of the mined rules.

---

## 📌 Project Objective

Analyze customer purchasing behavior by discovering products that are frequently
bought together. Generate association rules (Support, Confidence, Lift) that help
retailers improve **product recommendations, bundling, cross-selling, shelf
placement, inventory management, and marketing strategies.**

## 🗂️ Project Structure

```
BasketDNA/
├── data/
│   └── online+retail/
│       └── Online Retail.xlsx        # raw transactional dataset
├── notebook/
│   ├── Basket_DNA.ipynb              # full analysis: cleaning → EDA → Apriori → rules → viz
│   └── association_rules.csv         # exported rules used by the dashboard
├── app/
│   ├── streamlit_app.py              # interactive recommendation dashboard
│   ├── requirements.txt
│   ├── association_rules.csv
│   └── Online Retail.xlsx            # used to build the product search catalog
├── reports/                           # exported chart images from the notebook
├── requirements.txt                   # root env for running the notebook
└── README.md
```

## 🧠 Workflow

1. Load the Online Retail dataset (`.xlsx`).
2. **Clean:** drop missing values, drop duplicates, remove cancelled invoices
   (`InvoiceNo` starting with `'C'`), remove non-positive quantities.
3. **EDA:** shape, dtypes, missing values, top-selling products, unique
   invoices/products.
4. **Transform** into a basket matrix — rows = invoices, columns = products,
   values = 1/0 purchased.
5. **Apriori:** mine frequent itemsets at `min_support = 0.02`.
6. **Association rules:** generate at `min_confidence = 0.5`, with Support,
   Confidence, and Lift computed for each rule.
7. **Visualize:** top-selling products, top frequent itemsets, top rules, and
   a NetworkX rule-network graph.
8. **Export** `association_rules.csv` for downstream use.
9. **Serve** the rules through an interactive Streamlit dashboard (`/app`).

## 📊 Key Result (example rule)

| Antecedent | Consequent | Support | Confidence | Lift |
|---|---|---|---|---|
| ALARM CLOCK BAKELIKE GREEN | ALARM CLOCK BAKELIKE RED | 0.0286 | 67.2% | 14.2 |

A lift of **14.2** means these two products co-occur roughly 14x more often
than pure chance would predict — a strong, actionable cross-sell signal.

## 💡 Business Insights

- Color variants of the same product show strong purchasing relationships
  (e.g., GREEN alarm clocks → RED alarm clocks).
- Lift values well above 1 confirm these are genuine associations, not
  coincidental co-purchases.
- These rules directly support recommendation engines, product bundling,
  store layout/shelf placement, inventory planning, and targeted promotions.

## 🖥️ Running the Notebook

```bash
pip install -r requirements.txt
jupyter notebook notebook/Basket_DNA.ipynb
```

## 🚀 Running the Dashboard

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).
Pick items in the multiselect box and get live, explainable recommendations
(Support / Confidence / Lift shown per suggestion), plus a rule-network graph
and a full rules table.

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · mlxtend (Apriori) · NetworkX · Streamlit · openpyxl

## 🔭 Future Scope

- Mine rules per-country / per-season for regional and seasonal patterns.
- Factor in `UnitPrice` to estimate revenue uplift per recommended bundle.
- Try FP-Growth for faster mining at larger scale.
- Expose the recommendation logic as a FastAPI endpoint for integration into
  a real checkout flow.
- Segment-aware rules (e.g., high-value vs. low-value customer cohorts).

## 📄 License

For educational and portfolio use.
