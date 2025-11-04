import argparse
import json
import math
import os
import re
import warnings

import numpy as np
from mathruler.grader import extract_boxed_content, grade_answer

# -------------------------------------------------
# 1.  Config
# -------------------------------------------------
sub_question_list = {"prob_59", "prob_87", "prob_159", "prob_182", "prob_198", "prob_311", "prob_367", "prob_377", "prob_396", "prob_413", "prob_419", "prob_425", "prob_494", "prob_500", "prob_538", "prob_547", "prob_587", "prob_617", "prob_689", "prob_692", "prob_774", "prob_784", "prob_799", "prob_835", "prob_866", "prob_964", "prob_969", "prob_991", "prob_995", "prob_1091", "prob_1234", "prob_1255", "prob_1333", "prob_1401", "prob_1425", "prob_1463", "prob_1469", "prob_1488", "prob_1494", "prob_1498", "prob_1501", "prob_1554", "prob_1594", "prob_1613", "prob_1644", "prob_1729", "prob_1750", "prob_1790", "prob_1794", "prob_1850", "prob_1933", "prob_1973", "prob_1980", "prob_2053", "prob_2070", "prob_2091", "prob_2197", "prob_2202", "prob_2209", "prob_2264", "prob_2405", "prob_2457", "prob_2573", "prob_2636", "prob_2802", "prob_2805", "prob_2845", "prob_2908", "prob_2927", "prob_2934", "prob_2951", "prob_2964", "prob_2976", "prob_2989", "prob_3024", "prob_3028", "prob_3036", "prob_3048", "prob_3187", "prob_3399", "prob_3427", "prob_3470", "prob_3481", "prob_3509", "prob_3569", "prob_3604", "prob_3625", "prob_3689", "prob_3711", "prob_3719", "prob_3791", "prob_3831", "prob_3855", "prob_3924", "prob_3931", "prob_3964", "prob_3967", "prob_3975", "prob_4022", "prob_4040", "prob_4097", "prob_4133", "prob_4293", "prob_4299", "prob_4300", "prob_4333", "prob_4453", "prob_4458", "prob_4508", "prob_4527", "prob_4573", "prob_4651", "prob_4716", "prob_4826", "prob_4859", "prob_4937", "prob_5017", "prob_5084", "prob_5113", "prob_5176", "prob_5177", "prob_5211", "prob_5303", "prob_5314", "prob_5347", "prob_5422", "prob_5427", "prob_5489", "prob_5494", "prob_5548", "prob_5605", "prob_5712", "prob_5890", "prob_5900", "prob_5909", "prob_5932", "prob_5950", "prob_6001", "prob_6055", "prob_6113", "prob_6183", "prob_6194", "prob_6196", "prob_6206", "prob_6300", "prob_6318", "prob_6380", "prob_6388", "prob_6407", "prob_6426", "prob_6430", "prob_6442", "prob_6470", "prob_6472", "prob_6832", "prob_6943", "prob_6999", "prob_7005", "prob_7051", "prob_7065", "prob_7066", "prob_7069", "prob_7085", "prob_7092", "prob_7135", "prob_7146", "prob_7194", "prob_7236", "prob_7305", "prob_7365", "prob_7379", "prob_7396", "prob_7440", "prob_7478", "prob_7500", "prob_7544", "prob_7614", "prob_7622", "prob_7646", "prob_7716", "prob_7719", "prob_7748", "prob_7770", "prob_7778", "prob_7836", "prob_7923", "prob_7934", "prob_7946", "prob_7969", "prob_8022", "prob_8125", "prob_8162", "prob_8163", "prob_8187", "prob_8274", "prob_8313", "prob_8376", "prob_8398", "prob_8438", "prob_8468", "prob_8506", "prob_8567", "prob_8600", "prob_8635", "prob_8665", "prob_8679", "prob_8713", "prob_8749", "prob_8952", "prob_8994"}

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_ROOT = os.path.join(HERE, "results")
DEFAULT_ANSWER_DIR = "/proj/long-multi/kqian/granite-3b-ablate/kqian-tts_gyao/verl_dev/eval/pgps9k/results/new_9k_qwen3b_with_text_240_kl_280_without_text_200_4096_tmp_0.6"
DEFAULT_GT_FILE = os.path.join(HERE, "PGPS9K", "test.json")


# -------------------------------------------------
# 2.  CLI
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PGPS9K results")
    parser.add_argument(
        "--answer-dir",
        default=DEFAULT_ANSWER_DIR,
        help="Directory containing answer files (defaults to latest run)",
    )
    parser.add_argument(
        "--gt-file",
        default=DEFAULT_GT_FILE,
        help="Ground-truth JSON file path",
    )
    parser.add_argument(
        "--scan-all-results",
        action="store_true",
        help="Evaluate each subdirectory under the results root",
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing result subdirectories when scanning",
    )
    parser.add_argument(
        "--scan-output",
        default=None,
        help="Optional text file to capture --scan-all-results output",
    )
    return parser.parse_args()


# -------------------------------------------------
# 3.  Helpers
# -------------------------------------------------
def is_close_enough(t_input, b_input):
    if t_input == "None" or b_input == "None":
        return False
    try:
        if isinstance(t_input, str):
            expr = re.sub(r'\\sqrt{([^}]+)}', r'math.sqrt(\1)', t_input)
            expr = re.sub(r'(\d)(math\.sqrt)', r'\1*\2', expr)
            expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)
            expr = re.sub(r'\)(\s*\d)', r')*\1', expr)
            expr = re.sub(r'\)(\s*math\.sqrt)', r')*\1', expr)
            with warnings.catch_warnings():
                warnings.simplefilter("error", SyntaxWarning)
                t_val = eval(expr, {"__builtins__": {}}, {"math": math})
        else:
            t_val = float(t_input)
        b_val = float(b_input)
    except Exception:
        return False

    if t_val == b_val == 0:
        return True
    if b_val == 0:
        return False
    try:
        t_val - b_val
    except Exception:
        return False

    return abs(t_val - b_val) / abs(b_val) <= 0.01


def pass_at_1(total, correct):
    if total == 0:
        return float("nan")
    return 1 - np.prod(1 - 1 / np.arange(total - correct + 1, total + 1))


def show(tag, T, C, N, total_len=0, emit=print):
    emit(f"=== {tag} ===")
    if T:
        emit(f"Accuracy: {C / T:.4f}")
        emit(f"Total   : {T}")
        emit(f"Correct : {C}")
        emit(f"None    : {N}")
        emit(f"avg len: {total_len / T}")
        emit(f"Pass@1  : {pass_at_1(T, C):.4f}\n")
    else:
        emit("No data.\n")


# -------------------------------------------------
# 4.  Evaluation
# -------------------------------------------------
def evaluate_directory(answer_dir, gt_dict, label=None, emit=print):
    if label is not None:
        emit(f"Directory: {label}")

    tot = crr = none = 0
    sub_tot = sub_crr = sub_none = 0
    non_tot = non_crr = non_none = 0
    total_len = 0

    for fname in sorted(os.listdir(answer_dir)):
        fpath = os.path.join(answer_dir, fname)
        if not os.path.isfile(fpath):
            continue

        parts = fname.split("_")
        if len(parts) < 2:
            continue

        qname = "prob_" + parts[1]

        with open(fpath) as f:
            answer = f.read()
            total_len += len(answer)
            ans = extract_boxed_content(answer).replace("°", "")

        gt = gt_dict.get(qname)
        if gt is None:
            continue

        ok = grade_answer(ans, gt["answer"]) or is_close_enough(ans, gt["answer"])

        tot += 1
        if ok:
            crr += 1
        if ans == "None":
            none += 1

        if qname in sub_question_list:
            sub_tot += 1
            if ok:
                sub_crr += 1
            if ans == "None":
                sub_none += 1
        else:
            non_tot += 1
            if ok:
                non_crr += 1
            if ans == "None":
                non_none += 1

    show("Overall", tot, crr, none, total_len, emit=emit)
    show("In list", sub_tot, sub_crr, sub_none, emit=emit)
    show("Not in list", non_tot, non_crr, non_none, emit=emit)


# -------------------------------------------------
# 5.  Entrypoint
# -------------------------------------------------
def main():
    args = parse_args()

    answer_dir = args.answer_dir
    gt_file_path = args.gt_file
    results_root = args.results_root

    if not os.path.isabs(gt_file_path):
        gt_file_path = os.path.join(HERE, gt_file_path)

    if not os.path.exists(gt_file_path):
        raise FileNotFoundError(f"Ground-truth file not found: {gt_file_path}")

    with open(gt_file_path) as f:
        gt_dict = json.load(f)

    if args.scan_all_results:
        if not os.path.isabs(results_root):
            results_root = os.path.join(HERE, results_root)
        if not os.path.isdir(results_root):
            raise NotADirectoryError(f"Results root not found: {results_root}")

        subdirs = [d for d in sorted(os.listdir(results_root)) if os.path.isdir(os.path.join(results_root, d))]
        if not subdirs:
            print(f"No subdirectories found under {results_root}.")
            return

        output_handle = None
        if args.scan_output:
            output_path = args.scan_output
            if not os.path.isabs(output_path):
                output_path = os.path.join(results_root, output_path)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            output_handle = open(output_path, "w", encoding="utf-8")

            def emit(line=""):
                print(line)
                output_handle.write(line + "\n")
        else:
            def emit(line=""):
                print(line)

        try:
            for idx, subdir in enumerate(subdirs):
                full_path = os.path.join(results_root, subdir)
                evaluate_directory(full_path, gt_dict, label=subdir, emit=emit)
                if idx != len(subdirs) - 1:
                    emit()
        finally:
            if output_handle is not None:
                output_handle.close()
    else:
        if not os.path.isabs(answer_dir):
            answer_dir = os.path.join(HERE, answer_dir)
        if not os.path.isdir(answer_dir):
            raise NotADirectoryError(f"Answer directory not found: {answer_dir}")
        evaluate_directory(answer_dir, gt_dict)


if __name__ == "__main__":
    main()
