import openai
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import time

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def classify(report: str, api_key: str, instruct_prompt: str) -> str:
    full_prompt = f"""
    Task: {instruct_prompt}
    ---
    Report:
    {report}
    ---
    """
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ])
        
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error classifying report: {e}")
        return "error"

def main():
    data_path = 'labeled_data.csv'
    test_data = load_data(data_path)

    actual_injury_cause_labels = test_data['Injury Cause'].tolist()
    actual_root_cause_labels = test_data['Root Cause'].tolist()
    actual_severity_labels = test_data['Severity'].tolist()
    actual_body_part_labels = test_data['Body Part'].tolist()
    actual_accident_time_labels = test_data['Accident Time'].tolist()

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    openai.api_key = api_key
    
    reports = []

    injury_cause_labels = []
    root_cause_labels = []
    severity_labels = []
    body_part_labels = []
    accident_time_labels = []

    injury_cause_prompt = """Determine the Injury Cause of the accident in the report. Your answer should be strictly one of the following: 'Electrocution', 'Struck by', 'Fall', or 'Caught in/between' without any additional text or explanations."""
        
    root_cause_prompt = """Determine the Root Cause of the accident in the report. Your answer should be strictly one of the following: 'Struck by', 'Caught in/between', 'Fall', 'Electrocution', or 'Unspecified' without any additional text and explanations."""

    severity_prompt = """Determine the Severity of the incident in the report. Your answer should be strictly one of the following: 'Fatal' or 'Nonfatal' without any additional text or explanations."""

    body_part_prompt = """Determine the main Body Part affected in the accident. Provide only and strictly the main body part affected without any additional text, explanations, or multiple body parts. If the information is not available, say 'Unspecified'."""

    accident_time_prompt = """Determine the Accident Time of the accident in the report. The answer should strictly be in the format HH:MM am/pm. If the information is not available, say 'Unspecified'. Do not include the date and any other additional text and explanations."""

    BATCH_SIZE = 10
    for i in range(0, len(test_data), BATCH_SIZE):
        batch = test_data.iloc[i:i+BATCH_SIZE]
        print(f"Processing batch starting from index {i}")
        for _, row in batch.iterrows():
            report = row["Report"]
            injury_cause = classify(report, api_key, injury_cause_prompt)
            root_cause = classify(report, api_key, root_cause_prompt)
            severity = classify(report, api_key, severity_prompt)
            body_part = classify(report, api_key, body_part_prompt)
            accident_time = classify(report, api_key, accident_time_prompt)
            time.sleep(3)
            injury_cause_labels.append(injury_cause)
            root_cause_labels.append(root_cause)
            severity_labels.append(severity)
            body_part_labels.append(body_part)
            accident_time_labels.append(accident_time)
            reports.append(report)
        # time.sleep(6)

    # accuracy = accuracy_score(actual_labels, predicted_labels)
    # conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=all_classes)

    results_df = pd.DataFrame({
        'Actual Injury Cause': actual_injury_cause_labels,
        'Predicted Injury Cause': injury_cause_labels,
        'Actual Root Cause': actual_root_cause_labels,
        'Predicted Root Cause': root_cause_labels,
        'Actual Severity': actual_severity_labels,
        'Predicted Severity': severity_labels,
        'Actual Body Part': actual_body_part_labels,
        'Predicted Body Part': body_part_labels,
        'Actual Accident Time': actual_accident_time_labels,
        'Predicted Accident Time': accident_time_labels,
        'Report': reports
    })

    results_file_path = "classification_results.csv"
    results_df.to_csv(results_file_path, index=False)

    # print("Accuracy:", accuracy)
    # print("Confusion Matrix:\n", conf_matrix)
    # print("Results saved to:", results_file_path)

if __name__ == '__main__':
    main()
