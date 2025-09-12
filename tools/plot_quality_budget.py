#!/usr/bin/env python3
"""
Read a budget sweep CSV and generate PNG plots for:
- EM/F1 vs budget
- Avg think tokens vs budget
- Leakage vs budget
- ECE vs budget

Usage:
  python tools/plot_quality_budget.py --csv artifacts/budget_sweep_eval.csv --out artifacts
"""
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Path to budget sweep CSV (budget, em, f1, avg_think_tokens, leakage, ece, brier)')
    ap.add_argument('--out', default='artifacts', help='Output directory for PNGs')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd  # type: ignore
    except Exception:
        print('pandas not available; plotting requires pandas + matplotlib')
        return 1
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print('matplotlib not available; install it to enable plotting')
        return 1

    df = pd.read_csv(args.csv)
    if 'budget' not in df.columns:
        raise SystemExit('CSV missing budget column')
    # Ensure sorted by budget
    df = df.sort_values('budget')

    # Plot EM/F1
    plt.figure(figsize=(6,4))
    if 'em' in df.columns:
        plt.plot(df['budget'], df['em'], marker='o', label='EM')
    if 'f1' in df.columns:
        plt.plot(df['budget'], df['f1'], marker='o', label='F1')
    plt.xlabel('Budget cap')
    plt.ylabel('Quality')
    plt.title('Quality vs Budget')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'quality_vs_budget.png')
    plt.close()

    # Avg think tokens
    if 'avg_think_tokens' in df.columns:
        plt.figure(figsize=(6,4))
        plt.plot(df['budget'], df['avg_think_tokens'], marker='o')
        plt.xlabel('Budget cap')
        plt.ylabel('Avg think tokens')
        plt.title('Avg Think Tokens vs Budget')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'think_vs_budget.png')
        plt.close()

    # Leakage
    if 'leakage' in df.columns:
        plt.figure(figsize=(6,4))
        plt.plot(df['budget'], df['leakage'], marker='o')
        plt.xlabel('Budget cap')
        plt.ylabel('Leakage rate')
        plt.title('Leakage vs Budget')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'leakage_vs_budget.png')
        plt.close()

    # ECE
    if 'ece' in df.columns:
        plt.figure(figsize=(6,4))
        plt.plot(df['budget'], df['ece'], marker='o')
        plt.xlabel('Budget cap')
        plt.ylabel('ECE')
        plt.title('ECE vs Budget')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'ece_vs_budget.png')
        plt.close()

    print(json.dumps({'plots': {
        'quality_vs_budget': str(out_dir / 'quality_vs_budget.png'),
        'think_vs_budget': str(out_dir / 'think_vs_budget.png'),
        'leakage_vs_budget': str(out_dir / 'leakage_vs_budget.png'),
        'ece_vs_budget': str(out_dir / 'ece_vs_budget.png'),
    }}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

