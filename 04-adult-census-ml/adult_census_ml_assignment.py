"""
Adult Census Income Dataset - Complete ML Assignment
Tasks: Dataset Understanding | Data Cleaning | Feature Engineering |
       Model Building | Performance Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: DATASET UNDERSTANDING (10 Marks)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TASK 1: DATASET UNDERSTANDING")
print("=" * 70)

df = pd.read_csv('adult_census.csv')

print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n📋 Column Info:")
print(df.dtypes.to_string())
print(f"\n📈 Basic Statistics:")
print(df.describe().to_string())

print(f"\n🎯 Target Variable Distribution:")
vc = df['income'].value_counts()
print(vc)
print(f"   Class Ratio (>50K / <=50K): {vc['>50K']/vc['<=50K']:.2f}")

print(f"\n❓ Missing / Unknown Values ('?'):")
for col in df.select_dtypes(include='object').columns:
    n_q = (df[col] == '?').sum()
    if n_q > 0:
        print(f"   {col}: {n_q} ({n_q/len(df)*100:.1f}%)")

# Visualisation – Task 1
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Task 1: Dataset Understanding", fontsize=16, fontweight='bold')

# Income distribution
ax = axes[0, 0]
colors = ['#2196F3', '#FF5722']
bars = ax.bar(vc.index, vc.values, color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, vc.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Income Distribution', fontsize=13, fontweight='bold')
ax.set_ylabel('Count')
ax.set_ylim(0, max(vc.values)*1.15)

# Age distribution by income
ax = axes[0, 1]
for inc, color in zip(['>50K', '<=50K'], ['#2196F3', '#FF5722']):
    subset = df[df['income'] == inc]['age']
    ax.hist(subset, bins=30, alpha=0.6, label=inc, color=color)
ax.set_title('Age Distribution by Income', fontsize=13, fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.legend()

# Education level counts
ax = axes[0, 2]
edu_order = df['education'].value_counts().index[:10]
edu_counts = df['education'].value_counts().iloc[:10]
ax.barh(edu_order[::-1], edu_counts.values[::-1], color='#4CAF50')
ax.set_title('Top 10 Education Levels', fontsize=13, fontweight='bold')
ax.set_xlabel('Count')

# Hours per week distribution
ax = axes[1, 0]
ax.hist(df['hours-per-week'], bins=40, color='#9C27B0', alpha=0.7, edgecolor='white')
ax.axvline(40, color='red', linestyle='--', label='40 hrs (full-time)')
ax.set_title('Hours Per Week Distribution', fontsize=13, fontweight='bold')
ax.set_xlabel('Hours per Week')
ax.legend()

# Workclass distribution
ax = axes[1, 1]
wc = df['workclass'].value_counts()
ax.pie(wc.values[:6], labels=wc.index[:6], autopct='%1.1f%%',
       colors=plt.cm.Set3.colors[:6])
ax.set_title('Workclass Distribution', fontsize=13, fontweight='bold')

# Correlation heatmap (numeric)
ax = axes[1, 2]
num_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Numeric Feature Correlations', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task1_dataset_understanding.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Task 1 chart saved.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: DATA CLEANING (20 Marks)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TASK 2: DATA CLEANING")
print("=" * 70)

df_clean = df.copy()

# 2a. Replace '?' with NaN
before = len(df_clean)
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = df_clean[col].replace('?', np.nan)

print(f"\n🔧 Step 1 – '?' → NaN:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# 2b. Fill missing categoricals with mode
cat_cols_with_nan = df_clean.select_dtypes(include='object').columns[
    df_clean.select_dtypes(include='object').isnull().any()
]
for col in cat_cols_with_nan:
    mode_val = df_clean[col].mode()[0]
    df_clean[col] = df_clean[col].fillna(mode_val)
    print(f"   Filled '{col}' with mode: '{mode_val}'")

# 2c. Remove duplicates
dups = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
print(f"\n🔧 Step 2 – Duplicates removed: {dups}")
print(f"   Rows after dedup: {len(df_clean)}")

# 2d. Strip whitespace from categorical
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = df_clean[col].str.strip()

# 2e. Outlier detection & capping (IQR) for numeric
print(f"\n🔧 Step 3 – Outlier capping (IQR method):")
num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
for col in num_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = ((df_clean[col] < lo) | (df_clean[col] > hi)).sum()
    df_clean[col] = df_clean[col].clip(lo, hi)
    print(f"   {col}: {n_out} outliers capped [{lo:.0f}, {hi:.0f}]")

print(f"\n✅ Cleaned dataset: {df_clean.shape}")
print(f"   Remaining nulls: {df_clean.isnull().sum().sum()}")

# Visualisation – Task 2
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Task 2: Data Cleaning", fontsize=16, fontweight='bold')

# Before/after shape
ax = axes[0, 0]
stages = ['Original', 'After\nCleaning']
counts = [before, len(df_clean)]
bars = ax.bar(stages, counts, color=['#FF9800', '#4CAF50'], width=0.5, edgecolor='white')
for bar, val in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
            f'{val:,}', ha='center', fontsize=12, fontweight='bold')
ax.set_title('Dataset Size: Before vs After', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Rows')
ax.set_ylim(0, max(counts)*1.1)

# Missing values heatmap (before)
ax = axes[0, 1]
missing_before = df.isnull().sum()
missing_after = df_clean.isnull().sum()
miss_df = pd.DataFrame({'Before': (df == '?').sum(), 'After': missing_after})
miss_df = miss_df[miss_df['Before'] > 0]
x = range(len(miss_df))
width = 0.35
ax.bar([i-width/2 for i in x], miss_df['Before'], width, label='Before', color='#FF5722')
ax.bar([i+width/2 for i in x], miss_df['After'], width, label='After', color='#4CAF50')
ax.set_xticks(list(x))
ax.set_xticklabels(miss_df.index, rotation=20)
ax.set_title('Missing Values: Before vs After', fontsize=13, fontweight='bold')
ax.legend()

# Boxplots after cleaning
for i, col in enumerate(['age', 'hours-per-week', 'capital-gain']):
    ax = axes[0 if i < 3 else 1, i % 3 + (0 if i < 3 else 0)]
    ax = axes[1, i] if i < 3 else axes[1, i-3]
    if i < 3:
        ax = axes[1, i]
    parts = ax.violinplot(df_clean[col], showmedians=True, showmeans=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#2196F3')
        pc.set_alpha(0.7)
    ax.set_title(f'{col} (cleaned)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task2_data_cleaning.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Task 2 chart saved.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: FEATURE ENGINEERING (15 Marks)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TASK 3: FEATURE ENGINEERING")
print("=" * 70)

df_feat = df_clean.copy()

# 3a. Encode target
df_feat['income_binary'] = (df_feat['income'] == '>50K').astype(int)
print(f"✅ Target encoded: '>50K' → 1, '<=50K' → 0")

# 3b. Create new features
df_feat['age_group'] = pd.cut(df_feat['age'],
                               bins=[0,25,35,50,65,100],
                               labels=['Youth','Young Adult','Middle-aged','Senior','Elder'])
df_feat['is_married'] = df_feat['marital-status'].isin(
    ['Married-civ-spouse','Married-AF-spouse']).astype(int)
df_feat['capital_net'] = df_feat['capital-gain'] - df_feat['capital-loss']
df_feat['is_high_education'] = (df_feat['education-num'] >= 13).astype(int)
df_feat['overtime'] = (df_feat['hours-per-week'] > 40).astype(int)
df_feat['is_white_collar'] = df_feat['occupation'].isin(
    ['Exec-managerial','Prof-specialty','Tech-support','Adm-clerical']).astype(int)

print("✅ New features created:")
print("   age_group, is_married, capital_net, is_high_education, overtime, is_white_collar")

# 3c. Label encode all remaining categoricals
le = LabelEncoder()
cat_cols = df_feat.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['income']]
for col in cat_cols:
    df_feat[col] = le.fit_transform(df_feat[col].astype(str))
print(f"✅ Label-encoded {len(cat_cols)} categorical columns")

# 3d. Drop original income column
df_feat = df_feat.drop(columns=['income'])

# 3e. Feature selection via correlation with target
feature_cols = [c for c in df_feat.columns if c != 'income_binary']
# Encode age_group (categorical) before correlation
df_feat['age_group'] = le.fit_transform(df_feat['age_group'].astype(str))
correlations = df_feat[feature_cols].corrwith(df_feat['income_binary']).abs().sort_values(ascending=False)
print(f"\n📊 Top 10 Features by Correlation with Income:")
print(correlations.head(10).to_string())

# Select top features
top_features = correlations.head(15).index.tolist()

# Visualisation – Task 3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Task 3: Feature Engineering", fontsize=16, fontweight='bold')

# Feature importance (correlation)
ax = axes[0]
top10 = correlations.head(10)
colors_feat = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10)))
bars = ax.barh(top10.index[::-1], top10.values[::-1], color=colors_feat)
ax.set_title('Feature Correlation with Income', fontsize=13, fontweight='bold')
ax.set_xlabel('|Correlation|')
for bar, val in zip(bars, top10.values[::-1]):
    ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

# Capital net distribution
ax = axes[1]
for label, color in [(0,'#FF5722'), (1,'#2196F3')]:
    subset = df_feat[df_feat['income_binary']==label]['capital_net']
    ax.hist(subset, bins=40, alpha=0.6,
            label=['>50K','<=50K'][1-label], color=color)
ax.set_title('Capital Net by Income Class', fontsize=13, fontweight='bold')
ax.set_xlabel('Capital Net (gain - loss)')
ax.set_xlim(-5000, 30000)
ax.legend()

# Education vs income
ax = axes[2]
edu_income = df_clean.groupby('education-num')['income'].apply(
    lambda x: (x=='>50K').mean()).reset_index()
edu_income.columns = ['edu_num', 'pct_high']
ax.plot(edu_income['edu_num'], edu_income['pct_high']*100,
        'o-', color='#9C27B0', linewidth=2, markersize=8)
ax.fill_between(edu_income['edu_num'], edu_income['pct_high']*100, alpha=0.2, color='#9C27B0')
ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
ax.set_title('% Earning >50K by Education Level', fontsize=13, fontweight='bold')
ax.set_xlabel('Education Num (1=lowest, 16=highest)')
ax.set_ylabel('% Earning >50K')
ax.legend()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task3_feature_engineering.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Task 3 chart saved.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: MODEL BUILDING (30 Marks)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TASK 4: MODEL BUILDING – Classification Algorithms")
print("=" * 70)

X = df_feat[top_features]
y = df_feat['income_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\n📐 Train/Test Split: {len(X_train):,} / {len(X_test):,}")
print(f"   Features used: {len(top_features)}")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'KNN':                 KNeighborsClassifier(n_neighbors=7),
    'SVM':                 SVC(kernel='rbf', probability=True, random_state=42, C=1.0)
}

results = {}
print("\n🤖 Training models...\n")

for name, model in models.items():
    # SVM / LR / KNN need scaled features; Tree / RF do not
    needs_scale = name in ['Logistic Regression', 'KNN', 'SVM']
    Xtr = X_train_sc if needs_scale else X_train
    Xte = X_test_sc if needs_scale else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:,1]

    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
    }
    print(f"  ✅ {name:20s}  "
          f"Acc={results[name]['accuracy']:.4f}  "
          f"F1={results[name]['f1']:.4f}  "
          f"AUC={results[name]['roc_auc']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5: PERFORMANCE EVALUATION (15 Marks)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TASK 5: PERFORMANCE EVALUATION")
print("=" * 70)

# Results table
print("\n📊 Performance Summary Table:")
print(f"{'Algorithm':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1 Score':>9} {'ROC-AUC':>8}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
          f"{r['recall']:>8.4f} {r['f1']:>9.4f} {r['roc_auc']:>8.4f}")

best_name = max(results, key=lambda k: results[k]['f1'])
print(f"\n🏆 Best Model (by F1): {best_name}  (F1={results[best_name]['f1']:.4f})")

# ── Visualisation – Tasks 4 & 5 ──────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#F8F9FA')
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
fig.suptitle("Tasks 4 & 5: Model Building & Performance Evaluation",
             fontsize=17, fontweight='bold', y=0.98)

algo_names = list(results.keys())
short = ['LR', 'DT', 'RF', 'KNN', 'SVM']
metrics_data = {m: [results[a][m] for a in algo_names]
                for m in ['accuracy','precision','recall','f1','roc_auc']}

COLORS = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336']

# 1 ─ Grouped bar chart of all metrics
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(5)
width = 0.16
metric_labels = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']
metric_keys   = ['accuracy','precision','recall','f1','roc_auc']
bar_colors    = ['#42A5F5','#66BB6A','#FFA726','#AB47BC','#EF5350']
for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, bar_colors)):
    vals = [results[a][key] for a in algo_names]
    bars = ax1.bar(x + i*width, vals, width, label=label, color=color, alpha=0.85)
ax1.set_xticks(x + width*2)
ax1.set_xticklabels(short, fontsize=12)
ax1.set_ylim(0.5, 1.05)
ax1.set_title('All Metrics by Algorithm', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylabel('Score')
ax1.axhline(0.9, color='gray', linestyle=':', alpha=0.5)

# 2 ─ ROC curves
ax2 = fig.add_subplot(gs[0, 2])
for (name, r), color in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax2.plot(fpr, tpr, color=color, linewidth=2,
             label=f"{name.split()[0]} ({r['roc_auc']:.3f})")
ax2.plot([0,1],[0,1],'k--', alpha=0.4)
ax2.set_title('ROC Curves', fontsize=13, fontweight='bold')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(fontsize=8, loc='lower right')

# 3 ─ Confusion matrices
for i, (name, r) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, i % 3]) if i < 3 else fig.add_subplot(gs[2, i-3])
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['<=50K','>50K'], yticklabels=['<=50K','>50K'],
                cbar=False, linewidths=0.5)
    ax.set_title(f'{name}\nF1={r["f1"]:.3f}  AUC={r["roc_auc"]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

# 4 ─ F1 comparison (last subplot position)
ax_f1 = fig.add_subplot(gs[2, 2])
f1_vals = [results[a]['f1'] for a in algo_names]
bar_objs = ax_f1.bar(short, f1_vals, color=COLORS, edgecolor='white', linewidth=1.5)
for bar, val in zip(bar_objs, f1_vals):
    ax_f1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
               f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax_f1.set_ylim(0, 1.05)
ax_f1.set_title('F1 Score Comparison', fontsize=13, fontweight='bold')
ax_f1.set_ylabel('F1 Score')
best_idx = f1_vals.index(max(f1_vals))
bar_objs[best_idx].set_edgecolor('gold')
bar_objs[best_idx].set_linewidth(3)
ax_f1.set_facecolor('#FAFAFA')

plt.savefig('/mnt/user-data/outputs/task45_model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Tasks 4 & 5 chart saved.")

# ── Final Summary Table (formatted) ──────────────────────────────────────────
fig_table, ax_t = plt.subplots(figsize=(12, 3.5))
ax_t.axis('off')
fig_table.patch.set_facecolor('#1A1A2E')

headers = ['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
table_data = []
for name, r in results.items():
    table_data.append([
        name,
        f"{r['accuracy']:.4f}",
        f"{r['precision']:.4f}",
        f"{r['recall']:.4f}",
        f"{r['f1']:.4f}",
        f"{r['roc_auc']:.4f}",
    ])

tbl = ax_t.table(cellText=table_data, colLabels=headers,
                 loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.3, 2.2)

# Style header
for j in range(len(headers)):
    tbl[(0,j)].set_facecolor('#0D47A1')
    tbl[(0,j)].set_text_props(color='white', fontweight='bold')

# Highlight best row
best_idx_row = list(results.keys()).index(best_name) + 1
for j in range(len(headers)):
    tbl[(best_idx_row,j)].set_facecolor('#E8F5E9')
    tbl[(best_idx_row,j)].set_text_props(fontweight='bold')

# Alternate row colours
for i in range(1, len(table_data)+1):
    for j in range(len(headers)):
        if i != best_idx_row:
            tbl[(i,j)].set_facecolor('#F5F5F5' if i%2==0 else 'white')

ax_t.set_title('Performance Evaluation Summary — Adult Census Income Dataset',
               fontsize=14, fontweight='bold', pad=15, color='#0D47A1')

fig_table.tight_layout()
plt.savefig('/mnt/user-data/outputs/task5_summary_table.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✅ Summary table saved.")
print("\n🎉 All tasks complete!")
