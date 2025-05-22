import streamlit as st
import openai, os, json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Bronchiolitis Chatbot Demo")

# 1) user question
user_q = st.text_input("Ask a question about your data:")

# 2) prepare system + function spec
functions = [
    {
      "name": "compare_albuterol_usage",
      "description": "Compare albuterol usage before vs. after a given year",
      "parameters": {
        "type": "object",
        "properties": {
          "cutoff_year": {"type":"integer","description":"Year threshold"}
        },
        "required": ["cutoff_year"]
      }
    }
]

if st.button("Submit") and user_q:
    # 3) call OpenAI with function calling
    response = openai.ChatCompletion.create(
      model="gpt-4o-mini",
      messages=[
        {"role":"system","content":"You are a data-savvy assistant."},
        {"role":"user","content": user_q}
      ],
      functions=functions,
      function_call="auto"
    )
    msg = response.choices[0].message

    # 4) detect function call
    if msg.get("function_call"):
        name = msg["function_call"]["name"]
        args = json.loads(msg["function_call"]["arguments"])
        result = compare_albuterol_usage(**args)

        # 5) ask LLM to produce user-friendly writeup
        followup = openai.ChatCompletion.create(
          model="gpt-4o-mini",
          messages=[
            {"role":"system","content":"You are a data analyst who writes concise reports."},
            {"role":"user","content": user_q},
            {"role":"assistant","content": None, "function_call": {
                "name": name,
                "arguments": json.dumps(args)
            }},
            {"role":"function", 
             "name": name, 
             "content": json.dumps(result)}
          ]
        )
        st.markdown(followup.choices[0].message.content)
    else:
        st.write(msg.content)

