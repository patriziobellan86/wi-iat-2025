import json
import csv
import os
import re
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from itertools import combinations

P_VALUE_THRESHOLD = 0.05  # Statistical significance threshold


def get_answer_letter_in_sentence(sentence, answers):
    """
    Return the letter of the answer that appears in the given sentence.
    """
    sentence_lower = sentence.lower()

    for letter, answer in answers:
        if answer.lower() in sentence_lower:
            return letter

    return sentence


def normalize_answer(sentence, answers):
    """
    Normalize the answer by converting to lowercase and mapping letters to digits.
    """
    sentence = get_answer_letter_in_sentence(sentence, answers)
    # Define the mapping from letters to digits
    letter_to_digit = {
        'a': '0',
        'b': '1',
        'c': '2',
        'd': '3',
        'e': '4',
        'f': '5',
        'g': '6',
        'h': '7',
        'i': '8',
        'j': '9'
    }

    # Convert the answer to lowercase
    normalized = sentence.lower()

    # Map letters to digits if applicable
    return letter_to_digit.get(normalized, normalized)


def load_system_outputs(json_file):
    """Load system outputs from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_correct_answers(answer_file):
    """Load correct answers from a text file."""
    with open(answer_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def extract_answer(system_outputs, correct_answers):
    """Compute accuracy and perform McNemar's test for statistical significance."""
    total_count = len(system_outputs)

    if total_count == 0:
        return {}, {}

    gold_standard = []
    final_pred = []

    for idx, entry in enumerate(system_outputs.values()):
        correct_answer = correct_answers[idx] if idx < len(correct_answers) else ""
        query = entry.get("query", "").strip()
        # Use regex to capture answer choices
        matches = re.findall(r'\t([A-C])\) ([^\t]+)', query)

        # Convert to a list of tuples (letter, answer text)
        answers = [(letter, answer.strip()) for letter, answer in matches]

        # Final decision maker's answer
        system_answer = entry.get("final-decison-maker-answer", "").strip()

        gold_standard.append(normalize_answer(correct_answer, answers))
        final_pred.append(normalize_answer(system_answer, answers))

    return gold_standard, final_pred


def permutation_test(y_true, y_pred1, y_pred2, num_permutations=1000):
    # Calcola le F1 macro originali
    f1_model1 = f1_score(y_true, y_pred1, average='macro')
    f1_model2 = f1_score(y_true, y_pred2, average='macro')
    observed_diff = abs(f1_model1 - f1_model2)

    combined_preds = np.array([y_pred1, y_pred2])
    count = 0

    for _ in range(num_permutations):
        # Permuta le predizioni
        permuted_preds = np.random.permutation(combined_preds.T).T
        f1_perm1 = f1_score(y_true, permuted_preds[0], average='macro')
        f1_perm2 = f1_score(y_true, permuted_preds[1], average='macro')
        diff = abs(f1_perm1 - f1_perm2)

        if diff >= observed_diff:
            count += 1

    # Calcolo del p-value
    p_value = count / num_permutations
    return observed_diff, p_value


def write_precision_recall_f1_to_csv(csv_file, results):
    """Write results to a CSV file."""
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Collect all expert IDs
        all_expert_ids = sorted({eid for res in results for eid in res["expert_metrics"].keys()})

        # Write header
        header = ["Dataset", "Expert Model", "Final Decision Maker Precision", "Final Decision Maker Recall",
                  "Final Decision Maker F1", "Majority Vote Precision", "Majority Vote Recall", "Majority Vote F1"]
        for eid in all_expert_ids:
            header.append(f"Expert {eid} Precision")
        for eid in all_expert_ids:
            header.append(f"Expert {eid} Recall")
        for eid in all_expert_ids:
            header.append(f"Expert {eid} F1")
        writer.writerow(header)

        # Write data rows
        for res in results:
            row = [
                res["dataset"],
                res["expert_model"],
                f"{res['final_precision']:.2%}",
                f"{res['final_recall']:.2%}",
                f"{res['final_f1']:.2%}",
                f"{res['majority_precision']:.2%}",
                f"{res['majority_recall']:.2%}",
                f"{res['majority_f1']:.2%}"
            ]
            # Append expert accuracies
            row += [f"{res['expert_metrics'].get(eid, 0)[0]:.2%}" for eid in all_expert_ids]
            row += [f"{res['expert_metrics'].get(eid, 0)[1]:.2%}" for eid in all_expert_ids]
            row += [f"{res['expert_metrics'].get(eid, 0)[2]:.2%}" for eid in all_expert_ids]
            writer.writerow(row)


def get_system_output_paths(results_dir, gold_standard_dir, dataset_names):
    system_output_paths = []

    for dataset_name in dataset_names:
        # Define the path to the dataset's gold standard file
        gold_standard_file = os.path.join(gold_standard_dir, f"{dataset_name}.txt")

        # Traverse the results directory for the specified dataset
        dataset_results_dir = os.path.join(results_dir, dataset_name)
        for root, dirs, files in os.walk(dataset_results_dir):
            for file in files:
                if file == "queries_answers.json":
                    system_output_file = os.path.join(root, file)
                    system_output_paths.append((system_output_file, gold_standard_file))

    return system_output_paths


def main():
    results_dir = "Results/new_results/results"
    gold_standard_dir = "Results/gold_standard"
    dataset_names = ["socialsupport"]  # Add your dataset names here
    system_output_paths = get_system_output_paths(results_dir, gold_standard_dir, dataset_names)

    output_csv_file = "new_macro_f1_socialsupport.csv"
    results = []

    for system_output_file, correct_answer_file in system_output_paths:
        if not os.path.exists(system_output_file) or not os.path.exists(correct_answer_file):
            print(f"Skipping {system_output_file} or {correct_answer_file}: File not found.")
            continue

        system_outputs = load_system_outputs(system_output_file)
        correct_answers = load_correct_answers(correct_answer_file)

        gold_standard, final_pred = extract_answer(
            system_outputs, correct_answers)

        system_output_file_path = Path(system_output_file)
        system_output_file_parent = system_output_file_path.parent
        system_output_file_grandparent = system_output_file_parent.parent

        results.append({
            "dataset": system_output_file_grandparent.stem,
            "expert_model": system_output_file_parent.stem,
            "gold_standard": gold_standard,
            "final_pred": final_pred
        })
    for (result1, result2) in combinations(results, 2):
        observed_diff, p_value = permutation_test(result1["gold_standard"], result1["final_pred"], result2["final_pred"])
        significant = p_value < P_VALUE_THRESHOLD
        comparison_string = f"{result1['expert_model']} & {result2['expert_model']} & {observed_diff} & {p_value} & {'Yes' if significant else 'No'}  \\\\"
        comparison_string = comparison_string.replace("-1.2-0.9-1-Llama-3", "")
        print(comparison_string)



if __name__ == "__main__":
    main()
