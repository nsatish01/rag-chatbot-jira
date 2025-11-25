
import pandas as pd

# Load the CSV
jira_df = pd.read_csv(r"C:\\Users\\snagarajan\\AI_learnings\\rag-chatbot-jira\\data\\sample_jira_tickets.csv",
delimiter=",",
    quotechar='"',
    escapechar="\\",
    engine="python",
    on_bad_lines="skip"  # skips problematic rows
)

# Fill NaN values with empty strings for all columns
jira_df = jira_df.fillna('')

# Check the first few rows
print(jira_df.head())

# If you want to prepare for vector DB insertion:
# Convert each row into a dictionary (key = column name, value = cell content)
records = jira_df.to_dict(orient='records')

# Example: print one record
print(records[0])
