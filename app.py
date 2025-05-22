# streamlit_app.py
import os, json

import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
from scipy.stats import chi2_contingency, ttest_ind

# ── 1) Setup ────────────────────────────────────────────────────────────────────
openai.api_key = st.secrets["openai"]["api_key"]

@st.cache_data
def load_data(path="synthetic_bronchiolitis_dataset.csv"):
    df = pd.read_csv(path, parse_dates=["index_year"])
    return df

df = load_data()

# ── 2) Generic comparison function ─────────────────────────────────────────────
def compare_usage(column: str, cutoff_year: int, positive_value=None):
    # 1) pull out just the year
    df["index_year"] = df["index_year"].dt.year

    # 2) determine period
    df["period"] = df["index_year"].apply(
        lambda y: "after" if y >= cutoff_year else "before"
    )

    if positive_value is not None:
        # categorical χ² path…
        mask = df[column].apply(lambda x: x == positive_value 
                                         or (isinstance(x, str) and positive_value in x))
        df["__flag"] = mask.astype(int)
        ctab = pd.crosstab(df["period"], df["__flag"])
        # fill missing rows/cols…
        for p in ["before","after"]:
            if p not in ctab.index: ctab.loc[p] = [0,0]
        for val in [0,1]:
            if val not in ctab.columns: ctab[val] = 0
        chi2, pval, _, _ = chi2_contingency(ctab.sort_index().sort_index(axis=1))
        return {
            "contingency_table": ctab.sort_index().sort_index(axis=1).to_dict(),
            "chi2_statistic": float(chi2),
            "p_value": float(pval),
            "n_before": int(ctab.loc["before"].sum()),
            "n_after": int(ctab.loc["after"].sum())
        }

    else:
        # numeric t-test path…
        before = df.loc[df["period"]=="before", column].dropna()
        after  = df.loc[df["period"]=="after",  column].dropna()
        t_stat, pval_

# ── 3) Expose to OpenAI ────────────────────────────────────────────────────────
functions = [
    {
      "name": "compare_usage",
      "description": "Compare a column’s distribution before vs. after a given year",
      "parameters": {
        "type": "object",
        "properties": {
          "column": {
            "type": "string",
            "description": "Name of the dataframe column to compare"
          },
          "cutoff_year": {
            "type": "integer",
            "description": "Year threshold for before/after splitting"
          },
          "positive_value": {
            "type": ["string","null"],
            "description": (
              "For categorical columns: the value to treat as “positive” "
              "(e.g. 1 or 'dexamethasone'); omit for numeric t-test"
            )
          }
        },
        "required": ["column","cutoff_year"]
      }
    }
]

# ── 4) Streamlit UI ────────────────────────────────────────────────────────────
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


