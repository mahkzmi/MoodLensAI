#!/usr/bin/env python3
"""
MoodLens - Dataset Inspection Script

This script performs a complete inspection of the emotion dataset
before any preprocessing or modeling begins.

Usage:
    python inspect_dataset.py

The dataset is split into three files:
    - train.txt: Training data
    - val.txt: Validation data  
    - test.txt: Test data

Each file format: "text;label"
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter


def inspect_file(file_path: Path, split_name: str) -> dict:
    """
    Inspect a single data file.
    
    Args:
        file_path: Path to the .txt file
        split_name: Name of the split (Train/Val/Test)
    
    Returns:
        dict: Inspection results for this file
    """
    print(f"\n{'='*60}")
    print(f"📂 {split_name} SET")
    print(f"{'='*60}")
    print(f"File: {file_path.name}")
    
    # 1. بررسی وجود فایل
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return {}
    
    # 2. بارگذاری داده
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # جداسازی با ; (آخرین occurrence چون ممکنه متن شامل ; باشه)
                parts = line.rsplit(';', 1)
                if len(parts) == 2:
                    data.append({'text': parts[0], 'label': parts[1]})
    
    df = pd.DataFrame(data)
    
    print(f"\n📊 Basic Statistics:")
    print(f"  Total Rows: {len(df):,}")
    print(f"  Total Columns: {len(df.columns)}")
    print(f"  Columns: {', '.join(df.columns)}")
    
    # 3. بررسی برچسب‌ها
    print(f"\n🏷️ Label Analysis:")
    labels = df['label'].unique()
    print(f"  Unique Labels: {len(labels)}")
    print(f"  Labels: {sorted(labels)}")
    
    # توزیع برچسب‌ها
    label_counts = df['label'].value_counts()
    print(f"\n  Label Distribution:")
    for label, count in label_counts.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)  # بار چارت ساده
        print(f"    {label:12s}: {count:6,} ({pct:5.2f}%) {bar}")
    
    # تعادل داده
    max_count = label_counts.max()
    min_count = label_counts.min()
    ratio = max_count / min_count
    
    print(f"\n  Balance Check:")
    print(f"    Max Class: {max_count:,}")
    print(f"    Min Class: {min_count:,}")
    print(f"    Max/Min Ratio: {ratio:.2f}")
    
    if ratio < 1.5:
        print("    ✅ Well-balanced")
    elif ratio < 3:
        print("    ⚠️  Moderately imbalanced")
    else:
        print("    ❌ Highly imbalanced")
    
    # 4. بررسی متن‌ها
    print(f"\n📝 Text Analysis:")
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"  Text Length:")
    print(f"    Min: {df['text_length'].min():,}")
    print(f"    Max: {df['text_length'].max():,}")
    print(f"    Mean: {df['text_length'].mean():.2f}")
    print(f"    Median: {df['text_length'].median():.2f}")
    
    print(f"\n  Word Count:")
    print(f"    Min: {df['word_count'].min():,}")
    print(f"    Max: {df['word_count'].max():,}")
    print(f"    Mean: {df['word_count'].mean():.2f}")
    print(f"    Median: {df['word_count'].median():.2f}")
    
    # 5. نمونه‌های تصادفی
    print(f"\n📌 Sample Texts (5 random):")
    for idx, row in df.sample(5, random_state=42).iterrows():
        text = row['text']
        label = row['label']
        print(f"\n  [{label}]")
        print(f"  {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"  (Length: {len(text)}, Words: {row['word_count']})")
    
    # 6. داده‌های تکراری و گم‌شده
    print(f"\n🔍 Data Quality:")
    duplicates = df.duplicated().sum()
    print(f"  Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
    
    missing = df.isnull().sum().sum()
    print(f"  Missing values: {missing:,}")
    
    return {
        'split': split_name,
        'rows': len(df),
        'labels': sorted(labels),
        'label_counts': label_counts.to_dict(),
        'balance_ratio': ratio,
        'avg_text_length': df['text_length'].mean(),
        'avg_word_count': df['word_count'].mean(),
        'duplicates': duplicates,
        'missing': missing
    }


def generate_markdown_report(reports: dict, output_path: Path) -> None:
    """
    Generate a markdown report from inspection results.
    
    Args:
        reports: Dictionary of inspection results for each split
        output_path: Path to save the markdown file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# MoodLens - Dataset Inspection Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report provides a comprehensive analysis of the Kaggle Emotion Dataset.\n")
        f.write("The dataset is split into three files: Train, Validation, and Test.\n\n")
        
        # خلاصه
        f.write("### Summary\n\n")
        f.write("| Split | Rows | Classes | Balance Ratio | Avg Text Length |\n")
        f.write("|-------|------|---------|---------------|-----------------|\n")
        
        total_rows = 0
        all_labels = set()
        
        for name, rep in reports.items():
            if rep:
                total_rows += rep['rows']
                all_labels.update(rep['labels'])
                f.write(f"| {name} | {rep['rows']:,} | {len(rep['labels'])} | {rep['balance_ratio']:.2f} | {rep['avg_text_length']:.0f} |\n")
        
        f.write(f"\n**Total Rows:** {total_rows:,}\n")
        f.write(f"**Total Unique Labels:** {len(all_labels)}\n")
        f.write(f"**All Labels:** {', '.join(sorted(all_labels))}\n\n")
        
        # بررسی هر بخش
        for name, rep in reports.items():
            if not rep:
                continue
                
            f.write(f"## {name} Set\n\n")
            
            f.write("### Label Distribution\n\n")
            f.write("| Label | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            for label, count in rep['label_counts'].items():
                pct = (count / rep['rows']) * 100
                f.write(f"| {label} | {count:,} | {pct:.2f}% |\n")
            
            f.write(f"\n### Statistics\n\n")
            f.write(f"- **Rows:** {rep['rows']:,}\n")
            f.write(f"- **Classes:** {len(rep['labels'])}\n")
            f.write(f"- **Balance Ratio:** {rep['balance_ratio']:.2f}\n")
            f.write(f"- **Avg Text Length:** {rep['avg_text_length']:.0f} characters\n")
            f.write(f"- **Avg Word Count:** {rep['avg_word_count']:.1f} words\n")
            f.write(f"- **Duplicate Rows:** {rep['duplicates']:,}\n")
            f.write(f"- **Missing Values:** {rep['missing']:,}\n\n")
        
        # نتیجه‌گیری
        f.write("## Key Findings\n\n")
        
        # بررسی همه برچسب‌ها در همه مجموعه‌ها
        all_labels_found = set()
        for rep in reports.values():
            if rep:
                all_labels_found.update(rep['labels'])
        
        f.write(f"1. **Dataset contains {len(all_labels_found)} emotion classes:** {', '.join(sorted(all_labels_found))}\n")
        
        # بررسی تعادل
        ratios = [rep['balance_ratio'] for rep in reports.values() if rep]
        if ratios and max(ratios) < 1.5:
            f.write("2. **Dataset is well-balanced** across all splits\n")
        elif ratios and max(ratios) < 3:
            f.write("2. **Dataset is moderately imbalanced**\n")
        else:
            f.write("2. **Dataset is highly imbalanced** - may need techniques like class weighting\n")
        
        f.write(f"3. **Total dataset size:** {total_rows:,} samples\n")
        f.write("4. **Format:** Each sample is `text;label` with semicolon delimiter\n")
        f.write("5. **No preprocessing applied yet** - raw text needs cleaning\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Create configuration file (`config.py`)\n")
        f.write("2. Build DataLoader for loading all three splits\n")
        f.write("3. Implement text preprocessing pipeline\n")
        f.write("4. Train Logistic Regression model\n")


def main():
    """Main entry point."""
    print("="*60)
    print("MoodLens - Dataset Inspection")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # مسیرهای فایل‌ها
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "raw"
    
    files = {
        'Train': data_dir / "train.txt",
        'Val': data_dir / "val.txt",
        'Test': data_dir / "test.txt"
    }
    
    # بررسی وجود فایل‌ها
    missing_files = [name for name, path in files.items() if not path.exists()]
    if missing_files:
        print("❌ Missing files:")
        for name in missing_files:
            print(f"  - {name}.txt")
        print("\nPlease ensure all three files are in data/raw/")
        return
    
    # بررسی هر فایل
    reports = {}
    for name, path in files.items():
        reports[name] = inspect_file(path, name)
    
    # ایجاد پوشه experiments
    experiments_dir = base_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    
    # ذخیره گزارش مارک‌دون
    md_path = experiments_dir / "dataset_inspection.md"
    generate_markdown_report(reports, md_path)
    
    print(f"\n{'='*60}")
    print("✅ Inspection Complete!")
    print(f"{'='*60}")
    print(f"\n📄 Report saved to: {md_path}")
    print("\nSummary:")
    for name, rep in reports.items():
        if rep:
            print(f"  {name}: {rep['rows']:,} rows, {len(rep['labels'])} classes")
    
    print(f"\nTotal samples: {sum(rep['rows'] for rep in reports.values() if rep):,}")
    print(f"Total classes: {len(set().union(*[set(rep['labels']) for rep in reports.values() if rep]))}")


if __name__ == "__main__":
    main()