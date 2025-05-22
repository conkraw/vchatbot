# streamlit_app.py
import os, json

import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind
import difflib


def resolve_column(user_col: str) -> str:
    """Map user’s guess to the closest real column name."""
    if user_col in df.columns:
        return user_col
    match = difflib.get_close_matches(user_col, df.columns, n=1, cutoff=0.6)
    if match:
        return match[0]
    raise KeyError(f"Column '{user_col}' not found in data")

# ── 1) Setup ────────────────────────────────────────────────────────────────────
openai.api_key = st.secrets["openai"]["api_key"]

@st.cache_data
def load_data(path="synthetic_bronchiolitis_dataset.csv"):
    # Don’t parse dates—just read the year column
    df = pd.read_csv(path)
    # If your column is named something else (e.g. "year"), rename it:
    if "year" in df.columns and "index_year" not in df.columns:
        df.rename(columns={"year": "index_year"}, inplace=True)
    # Now ensure it's integer
    df["index_year"] = df["index_year"].astype(int)
    return df


df = load_data()

def compare_usage(column: str, cutoff_year: int, positive_value=None):


    real_col = resolve_column(column)
    df["period"] = df["index_year"].apply(
        lambda y: "after" if y >= cutoff_year else "before"
    )

    # detect categorical string column
    if positive_value is not None or df[real_col].dtype == object:
        # if no positive_value given, count ANY non-empty string
        if positive_value is None:
            mask = (
                df[real_col].notnull()
                & df[real_col].astype(str).str.strip().ne("")
            )
        else:
            mask = df[real_col].astype(str).str.contains(positive_value)

        df["__flag"] = mask.astype(int)
        ctab = pd.crosstab(df["period"], df["__flag"])
        for p in ["before","after"]:
            if p not in ctab.index: ctab.loc[p] = [0,0]
        for val in [0,1]:
            if val not in ctab.columns: ctab[val] = 0
        ctab = ctab.sort_index().sort_index(axis=1)

        # ==== insert the rename here ====
        flag_name     = positive_value if isinstance(positive_value, str) else real_col
        use_label     = f"{flag_name}"
        not_use_label = f"No {flag_name}"
        ctab = ctab.rename(columns={0: not_use_label, 1: use_label})
        # ================================
        try:
            chi2, pval, _, _ = chi2_contingency(ctab, correction=False)
            test = "chi-square"
        except ValueError:
            # Fisher’s Exact for 2×2
            _, pval = fisher_exact(ctab)
            chi2 = None
            test = "fisher-exact"

        return {
            "test": test,
            "chi2_statistic": chi2,
            "p_value": pval,
            "contingency_table": ctab.to_dict(),
            "n_before": int(ctab.loc["before"].sum()),
            "n_after":  int(ctab.loc["after"].sum()),
        }
    # 3b) Numeric branch: two‐sample t-test
    else:
        before = df.loc[df["period"] == "before", real_col].dropna()
        after  = df.loc[df["period"] == "after",  real_col].dropna()
        t_stat, pval = ttest_ind(before, after, nan_policy="omit")

        return {
            "mean_before": float(before.mean()),
            "mean_after":  float(after.mean()),
            "n_before":    int(before.count()),
            "n_after":     int(after.count()),
            "t_statistic": float(t_stat),
            "p_value":     float(pval),
        }

# ── 3) Expose to OpenAI ────────────────────────────────────────────────────────
COLUMN_NAMES = df.columns.tolist()

functions = [
  {
    "name": "compare_usage",
    "description": "Compare a column’s distribution before vs. after a given year",
    "parameters": {
      "type": "object",
      "properties": {
        "column": {
          "type": "string",
          "enum": COLUMN_NAMES,
          "description": "One of: " + ", ".join(COLUMN_NAMES)
        },
        "cutoff_year": {"type": "integer"},
        "positive_value": {
          "type": ["string","null"],
          "description": "…"
        }
      },
      "required": ["column","cutoff_year"]
    }
  }
]


# ── 4) Streamlit UI ────────────────────────────────────────────────────────────
df = load_data("synthetic_bronchiolitis_dataset.csv")

# ← Put it here, just after df is ready
st.sidebar.subheader("Available columns")
st.sidebar.write(df.columns.tolist())

st.title("🔍 Bronchiolitis Chatbot")

user_q = st.text_input("Ask a question about your data…", 
                       placeholder="e.g. Is albuterol usage higher after 2022?")

if st.button("Submit") and user_q:
    # Ask LLM (with function schema)
    chat = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a data-savvy assistant."},
            {"role":"user","content":user_q}
        ],
        functions=functions,
        function_call="auto"
    )
    msg = chat.choices[0].message

    # If LLM calls our function, run it
    if msg.get("function_call"):
        args   = json.loads(msg["function_call"]["arguments"])
        result = compare_usage(**args)

        # Display raw results
        if "contingency_table" in result:
            ctab = pd.DataFrame(result["contingency_table"])
            st.subheader("Contingency Table")
            st.table(ctab)
        else:
            st.subheader("Group Statistics")
            st.write(f"Mean before: {result['mean_before']:.2f} "
                     f"(n={result['n_before']})")
            st.write(f"Mean after:  {result['mean_after']:.2f} "
                     f"(n={result['n_after']})")

        # Let LLM craft a human‐friendly summary
        followup = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system",
                 "content":"You are a data analyst who writes concise reports."},
                {"role":"user","content":user_q},
                {"role":"assistant","content":None,
                 "function_call": {
                   "name": msg["function_call"]["name"],
                   "arguments": msg["function_call"]["arguments"]
                 }},
                {"role":"function",
                 "name": msg["function_call"]["name"],
                 "content": json.dumps(result)}
            ]
        )
        st.markdown(followup.choices[0].message.content)

    else:
        # LLM answered directly
        st.write(msg.content)


