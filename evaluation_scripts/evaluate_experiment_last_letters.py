import json
import csv
import os
from collections import Counter
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path

P_VALUE_THRESHOLD = 0.05  # Statistical significance threshold


def normalize_answer(answer):
    """
    Normalize the answer by converting to lowercase and mapping letters to digits.
    """
    # Convert the answer to lowercase
    normalized = answer.lower()

    # Remove - in between letters
    normalized = normalized.replace("-", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("\'", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.split("/")[0]

    # Map letters to digits if applicable
    return normalized.strip()


def load_system_outputs(json_file):
    """Load system outputs from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_correct_answers(answer_file):
    """Load correct answers from a text file."""
    with open(answer_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def compute_accuracy_and_significance(system_outputs, correct_answers):
    """Compute accuracy and perform McNemar's test for statistical significance."""
    total_count = len(system_outputs)

    if total_count == 0:
        return 0, 0, {}, {}

    final_correct = []
    majority_correct = []
    expert_correct = {}

    majority_vote_correct_count = 0
    expert_correct_counts = Counter()
    expert_total_counts = Counter()

    for idx, entry in enumerate(system_outputs.values()):
        correct_answer = correct_answers[idx] if idx < len(correct_answers) else ""
        correct_answer_norm = correct_answer

        # Final decision maker's answer
        system_answer = entry.get("final-decison-maker-answer", "").strip()
        system_answer_norm = normalize_answer(system_answer)
        # if len(system_answer_norm) != len(correct_answer_norm):
        #     total_count = total_count - 1
        #     print(
        #         f"Skipping '{idx}' evaluation for system output: '{system_answer_norm}' (not comparable with '{correct_answer_norm}')")
        #     continue
        final_correct.append(system_answer_norm == correct_answer_norm)

        # Experts' answers
        expert_answers = []
        for expert in entry.get("query_answers", []):
            expert_id = expert.get("expert_ID")
            expert_answer_string = str(expert.get("final_answer", ""))
            expert_answer = expert_answer_string.strip()
            expert_answer_norm = normalize_answer(expert_answer)

            if expert_answer_norm:
                expert_total_counts[expert_id] += 1
                expert_correct.setdefault(expert_id, []).append(expert_answer_norm == correct_answer_norm)
                if expert_answer_norm == correct_answer_norm:
                    expert_correct_counts[expert_id] += 1

            expert_answers.append(expert_answer_norm)

        # Majority vote
        if expert_answers:
            most_common_answer, _ = Counter(expert_answers).most_common(1)[0]
            majority_correct.append(most_common_answer == correct_answer_norm)
            if most_common_answer == correct_answer_norm:
                majority_vote_correct_count += 1
        else:
            majority_correct.append(False)

    # Compute accuracies
    final_accuracy = sum(final_correct) / total_count
    majority_vote_accuracy = majority_vote_correct_count / total_count
    expert_accuracies = {eid: (expert_correct_counts[eid] / expert_total_counts[eid]) for eid in expert_total_counts}

    # Perform McNemar's test for significance
    significance_results = {}

    def mcnemar_test(ref_correct, other_correct):
        table = [[0, 0], [0, 0]]
        for ref, other in zip(ref_correct, other_correct):
            table[ref][other] += 1
        result = mcnemar(table, exact=True)
        return result.pvalue

    # Compare final decision maker vs. majority vote
    p_value = mcnemar_test(final_correct, majority_correct)
    significance_results["Majority Vote"] = {
        "p_value": p_value,
        "significant": p_value < P_VALUE_THRESHOLD
    }

    # Compare final decision maker vs. each expert
    for eid, expert_answers in expert_correct.items():
        p_value = mcnemar_test(final_correct, expert_answers)
        significance_results[f"Expert {eid}"] = {
            "p_value": p_value,
            "significant": p_value < P_VALUE_THRESHOLD
        }

    return final_accuracy, majority_vote_accuracy, expert_accuracies, significance_results


def write_accuracies_to_csv(csv_file, results):
    """Write results to a CSV file."""
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Collect all expert IDs
        all_expert_ids = sorted({eid for res in results for eid in res["expert_accuracies"].keys()})

        # Write header
        header = ["Dataset", "Expert Model", "Final Decision Maker Accuracy", "Majority Vote Accuracy"]
        for eid in all_expert_ids:
            header.append(f"Expert {eid} Accuracy")
        header.append("Majority Vote p-value")
        header.append("Majority Vote Significant")
        for eid in all_expert_ids:
            header.append(f"Expert {eid} p-value")
            header.append(f"Expert {eid} Significant")
        writer.writerow(header)

        # Write data rows
        for res in results:
            row = [
                res["dataset"],
                res["expert_model"],
                f"{res['final_accuracy']:.2%}",
                f"{res['majority_vote_accuracy']:.2%}"
            ]
            # Append expert accuracies
            row += [f"{res['expert_accuracies'].get(eid, 0):.2%}" for eid in all_expert_ids]
            # Append significance results for majority vote
            mv_result = res["significance_results"].get("Majority Vote", {})
            row.append(f"{mv_result.get('p_value', 'N/A'):.4f}")
            row.append("Yes" if mv_result.get("significant", False) else "No")
            # Append significance results for each expert
            for eid in all_expert_ids:
                exp_result = res["significance_results"].get(f"Expert {eid}", {})
                row.append(f"{exp_result.get('p_value', 'N/A'):.4f}")
                row.append("Yes" if exp_result.get("significant", False) else "No")
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
    dataset_names = ["last_letters"]  # Add your dataset names here
    system_output_paths = get_system_output_paths(results_dir, gold_standard_dir, dataset_names)

    output_csv_file = "new_accuracy_last_letters.csv"
    results = []

    for system_output_file, correct_answer_file in system_output_paths:
        if not os.path.exists(system_output_file) or not os.path.exists(correct_answer_file):
            print(f"Skipping {system_output_file} or {correct_answer_file}: File not found.")
            continue

        system_outputs = load_system_outputs(system_output_file)
        correct_answers = load_correct_answers(correct_answer_file)

        final_accuracy, majority_vote_accuracy, expert_accuracies, significance_results = compute_accuracy_and_significance(
            system_outputs, correct_answers)

        system_output_file_path = Path(system_output_file)
        system_output_file_parent = system_output_file_path.parent
        system_output_file_grandparent = system_output_file_parent.parent

        sample_name = system_output_file_grandparent.stem + ' ' + system_output_file_parent.stem

        results.append({
            "dataset": system_output_file_grandparent.stem,
            "expert_model": system_output_file_parent.stem,
            "final_accuracy": final_accuracy,
            "majority_vote_accuracy": majority_vote_accuracy,
            "expert_accuracies": expert_accuracies,
            "significance_results": significance_results
        })

    write_accuracies_to_csv(output_csv_file, results)
    print(f"Results saved in {output_csv_file}")


if __name__ == "__main__":
    main()
