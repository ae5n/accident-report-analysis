import openai
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def classify(report: str, api_key: str) -> str:
    instruct_prompt = f"""
    Task: Classify the provided construction accident report. Determine the cause of the accident, which may be: electrocution, struck, fall, or caught. Assess whether the incident was fatal or nonfatal.

    Your response should be exactly one of the following options, without additional explanations:
    - electrocution_nonfatal
    - struck_nonfatal
    - fall_nonfatal
    - caught_nonfatal
    - electrocution_fatal
    - struck_fatal
    - caught_fatal
    - fall_fatal
    ---
    Report:
    {report}
    ---
    """
    
    completion = openai.Completion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruct_prompt}
        ])
    
    return completion.choices[0].message['content']

def main():
    all_classes = [
        'electrocution_nonfatal',
        'struck_nonfatal',
        'fall_nonfatal',
        'caught_nonfatal',
        'electrocution_fatal',
        'struck_fatal',
        'caught_fatal',
        'fall_fatal'
    ]

    data_path = 'Data/cleaned_data_25.csv'
    test_data = load_data(data_path)

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    openai.api_key = api_key

    actual_labels = []
    predicted_labels = []
    for _, row in test_data.iterrows():
        report = row[1]
        actual_label = row[0]
        predicted_label = classify(report, api_key)
        actual_labels.append(actual_label)
        predicted_labels.append(predicted_label)

    accuracy = accuracy_score(actual_labels, predicted_labels)
    conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=all_classes)

    results_df = pd.DataFrame({
        'Actual Label': actual_labels,
        'Predicted Label': predicted_labels,
        'Report': test_data['Report']
    })
    results_file_path = "classification_results.csv"
    results_df.to_csv(results_file_path, index=False)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Results saved to:", results_file_path)

if __name__ == '__main__':
    main()
