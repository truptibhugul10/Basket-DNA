"""
Basket DNA — Interactive Product Recommendation Dashboard
-----------------------------------------------------------
Loads pre-computed Apriori association rules and lets a user pick
items they want to buy, then recommends what to add to the cart —
"customers who bought this also bought..." style, powered by
Support / Confidence / Lift instead of a black-box model.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Expected files in the same folder (or update the paths below):
    - association_rules.csv   (output of your notebook's Apriori pipeline)
    - Online Retail.xlsx      (optional — only used to build a full product
                               catalog for the search box; if missing, the
                               catalog falls back to products that appear
                               in the rules themselves)
"""

import ast
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Basket DNA — Smart Product Recommender",
    page_icon="🧺",
    layout="wide",
)

RULES_PATH = os.path.join(os.path.dirname(__file__), "association_rules.csv")
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "Online Retail.xlsx")


# ----------------------------------------------------------------------
# Data loading (cached so Apriori-derived rules aren't re-parsed on
# every click — this is the key perf trick for a Streamlit app)
# ----------------------------------------------------------------------
@st.cache_data
def load_rules(path: str) -> pd.DataFrame:
    rules = pd.read_csv(path)
    # antecedents/consequents are stored as string representations of
    # frozensets (e.g. "frozenset({'ALARM CLOCK BAKELIKE GREEN'})") —
    # convert them back into real Python sets for fast subset checks.
    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: set(ast.literal_eval(x.replace("frozenset(", "").replace(")", "")))
    )
    rules["consequents"] = rules["consequents"].apply(
        lambda x: set(ast.literal_eval(x.replace("frozenset(", "").replace(")", "")))
    )
    # strip stray whitespace that's common in this dataset's product names
    rules["antecedents"] = rules["antecedents"].apply(lambda s: {i.strip() for i in s})
    rules["consequents"] = rules["consequents"].apply(lambda s: {i.strip() for i in s})
    return rules


@st.cache_data
def load_catalog(rules: pd.DataFrame, raw_path: str):
    """Build the full product list for the search/multiselect box.
    Falls back gracefully if the raw Excel file isn't shipped with the app."""
    if os.path.exists(raw_path):
        df = pd.read_excel(raw_path)
        df = df.dropna(subset=["Description"])
        products = sorted(df["Description"].str.strip().unique().tolist())
        top_sellers = (
            df[df["Quantity"] > 0]
            .groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .index.str.strip()
            .tolist()
        )
        return products, top_sellers

    # Fallback: only products that appear somewhere in the rules
    all_items = set()
    for s in rules["antecedents"]:
        all_items |= s
    for s in rules["consequents"]:
        all_items |= s
    products = sorted(all_items)
    return products, products[:10]


rules = load_rules(RULES_PATH)
catalog, top_sellers = load_catalog(rules, RAW_DATA_PATH)


# ----------------------------------------------------------------------
# Recommendation engine
# ----------------------------------------------------------------------
def recommend(cart: list, rules: pd.DataFrame, min_conf: float, min_lift: float, top_n: int):
    cart_set = set(cart)
    if not cart_set:
        return pd.DataFrame()

    mask = rules["antecedents"].apply(lambda a: a.issubset(cart_set) and not a.isdisjoint(cart_set))
    matches = rules[mask].copy()

    # Don't recommend something already in the cart
    matches["consequents"] = matches["consequents"].apply(lambda c: c - cart_set)
    matches = matches[matches["consequents"].apply(len) > 0]

    matches = matches[(matches["confidence"] >= min_conf) & (matches["lift"] >= min_lift)]
    matches = matches.sort_values(by=["lift", "confidence"], ascending=False)
    return matches.head(top_n)


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Recommendation Settings")
min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)
min_lift = st.sidebar.slider("Minimum Lift", 0.0, float(max(rules["lift"].max(), 1)), 1.0, 0.5)
top_n = st.sidebar.slider("Number of recommendations", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Loaded **{len(rules)}** association rules across **{len(catalog)}** products."
)

# ----------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------
st.title("🧺 Basket DNA — Smart Product Recommender")
st.caption(
    "Pick the items you're planning to buy. The engine looks up Apriori "
    "association rules (Support / Confidence / Lift) mined from real "
    "purchase history to suggest what pairs well with your cart."
)

cart = st.multiselect(
    "🛒 What are you adding to your cart?",
    options=catalog,
    placeholder="Start typing a product name...",
)

col1, col2 = st.columns([2, 1])

with col1:
    if cart:
        st.subheader("✨ Recommended for you")
        recs = recommend(cart, rules, min_conf, min_lift, top_n)

        if recs.empty:
            st.info(
                "No strong association rules matched your cart at the current "
                "thresholds — here are our overall best sellers instead:"
            )
            for p in top_sellers:
                st.write(f"• {p}")
        else:
            for _, row in recs.iterrows():
                items = ", ".join(sorted(row["consequents"]))
                st.markdown(
                    f"**{items}**  \n"
                    f"because you have *{', '.join(sorted(row['antecedents']))}* "
                    f"— confidence **{row['confidence']*100:.1f}%**, "
                    f"lift **{row['lift']:.2f}**, support **{row['support']:.4f}**"
                )
                st.divider()
    else:
        st.info("Add at least one product above to see recommendations.")

with col2:
    st.subheader("🕸️ Rule Network")
    top_rules_for_graph = rules.sort_values("confidence", ascending=False).head(12)
    G = nx.DiGraph()
    for _, row in top_rules_for_graph.iterrows():
        a = ", ".join(row["antecedents"])
        c = ", ".join(row["consequents"])
        G.add_edge(a, c, weight=row["confidence"])

    fig, ax = plt.subplots(figsize=(5, 5))
    pos = nx.spring_layout(G, seed=42, k=0.9)
    nx.draw_networkx_nodes(G, pos, node_color="#ffb703", node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=12, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    ax.axis("off")
    st.pyplot(fig)

st.markdown("---")
with st.expander("📊 View all association rules"):
    st.dataframe(
        rules[["antecedents", "consequents", "support", "confidence", "lift"]]
        .sort_values("lift", ascending=False)
        .reset_index(drop=True)
    )
