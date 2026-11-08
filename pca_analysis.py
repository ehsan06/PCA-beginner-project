






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

df = pd.read_csv("cancer.csv")
print(df ,df.shape)
print(df.dtypes)

df.drop(['id','diagnosis','Unnamed: 32'],axis =1, inplace = True)


print(df.shape)
#
# print(df)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns ,index=df.index)
# print(df_scaled)

pca_full = PCA(n_components=None, svd_solver='auto', random_state=0)
pca_full.fit(df_scaled) #محاسبات مولفه های اصلی و محاسبه ی واریانس هر مولفه ی اصلی
explained = pca_full.explained_variance_ratio_
c_explained = np.cumsum(explained)
print('explained_variance_ratio: ','\n',explained,"\n",'c_explained :','\n', c_explained)

plt.figure(figsize=(8,4))
plt.plot(np.arange(1, len(explained)+1), explained, marker='o', label='explained variance ratio')
plt.plot(np.arange(1, len(explained)+1), c_explained, marker='x', label='cumulative')
plt.xlabel('Principal component')
plt.xticks(np.arange(1, len(explained)+1))
plt.ylabel('Proportion of variance')
plt.legend()
plt.title('Scree plot & cumulative explained variance')
plt.grid(True)
plt.tight_layout()
plt.show()

for i, (e, c) in enumerate(zip(explained, c_explained), start=1):
    print(f"PC{i:02d}: explained={e:.4f}, cumulative={c:.4f}")

k = np.argmax(c_explained >= 0.90) + 1

if k == 1 and c_explained[0] < 0.90:
    k = np.argmax(c_explained >= 0.80) + 1
print("chosen n_components:", k)

pca = PCA(n_components=k, random_state=0)
scores = pca.fit_transform(df_scaled)           # What it does: You are creating a new PCA object that only keeps the top 7 components.
                                                # fit_transform does two things: it calculates the directions (fit) and then
                                                # projects your 569 patients onto those 7 directions (transform).

loadings = pd.DataFrame(pca.components_.T,
                        columns=[f"PC{i+1}" for i in range(k)],
                        index=df_scaled.columns)                 #pca.components_ are the Eigenvectors. The .T (transpose) flips the table so that
                                                                 #your 30 original features are the rows and your 7 PCs are the columns.

scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(k)], index=df.index)
                                                                #what it does: It takes the coordinates of your 569 patients in the new
                                                                # 7D space and puts them in a clean DataFrame.

loadings.to_csv("pca_loadings.csv")
scores_df.to_csv("pca_scores.csv")


# 10) top contributing features per PC
for pc in loadings.columns:
    sorted_loads = loadings[pc].abs().sort_values(ascending=False)
    print("\nTop features for", pc)
    print(sorted_loads.head(10))

########################################################################################################################

original_df= pd.read_csv("cancer.csv")
target_labels = original_df['diagnosis']

sns.set_style("whitegrid")
plt.figure(figsize=(14, 12)) # اندازه ی صفحه را بزرگ انتخاب میکنیم برای خوانایی بهتر

sns.scatterplot(x='PC1', y='PC2',
                hue=target_labels,
                palette={'M': '#FF6347', 'B': '#4682B4'}, # کد رنگ قرمز (M), کد رنگ آبی (B)
                data=scores_df,
                s=60,         # Marker size
                alpha=0.7,    # میزان شفافیت برای اینکه نقطه هایی که روی هم میافتند قابل دیدن باشند.
                edgecolor='k' # لبه ی سیاه برای داده نقه ها
                )
# --- 3. Plot the Loadings (The Arrows - Features) ---
# مقادیر باگذاری های متغیر ها معمولا بین -1 و 1 میباشد.scores ها معمولا بسیار بزرگترند
# به arrow multiplier یا بزرگ کننده ی فلش ها نیاز داریم که در صفحه قابل دیدن باشند.
arrow_multiplier = 16

# Loop through every feature in the loadings DataFrame
for i, feature_name in enumerate(loadings.index):
    # مختصات برداری PC1 و PC2 را میگیریم
    x_vec = loadings.loc[feature_name, 'PC1'] * arrow_multiplier
    y_vec = loadings.loc[feature_name, 'PC2'] * arrow_multiplier

    # فلش های جهت دار را از نقطه ی (0و0) تا مختصات بردار ها میکشیم
    plt.arrow(0, 0, x_vec, y_vec,
              color='gray',
              alpha=0.8, # اندازه ی شفافیت فلش که کمی شفاف باشد
              head_width=0.1, # اندازه ی سر فلش
              linewidth=1.5)

    # نام متغیر ها در انتهای فلش ها قرار میدیم
    # ضرب در 1.1 میکنیم که کمی از انتهای فلش فاصله بگیرد
    plt.text(x_vec * 1.15, y_vec * 1.15,
             feature_name,
             color='black',
             ha='center', va='center', fontsize=10, weight='bold')

# خطوطی که از مرکز میگذرند
plt.axhline(0, color='grey', lw=1, ls='--')
plt.axvline(0, color='grey', lw=1, ls='--')

# نام گذاری محور ها توسط میزان واریانسی که توضیح میدهند
plt.xlabel(f'Principal Component 1 ({explained[0]*100:.1f}% Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({explained[1]*100:.1f}% Variance)', fontsize=12)
plt.title('PCA Biplot: Wisconsin Breast Cancer Data (PC1 vs PC2)', fontsize=16)

plt.legend(title='Diagnosis', loc='upper left', bbox_to_anchor=(1, 1)) # Move legend outside


plt.tight_layout()
plt.show()











# 11) Scatterplot of first two PCs (if k>=2)
# if k >= 2:
#     plt.figure(figsize=(6,5))
#     plt.scatter(scores_df["PC1"], scores_df["PC2"], s=20, alpha=0.7)
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.title("Projection on first two principal components")
#     plt.axhline(0, color='grey', linewidth=0.5)
#     plt.axvline(0, color='grey', linewidth=0.5)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#     # 12) Biplot (approx): plots scores and arrows for loadings on PC1 & PC2
#     if k >= 2:
#         plt.figure(figsize=(7, 6))
#         plt.scatter(scores_df["PC1"], scores_df["PC2"], alpha=0.6, s=15)
#         # scale loadings for arrows
#         coeffs = loadings.loc[:, ["PC1", "PC2"]]
#         for i, col in enumerate(coeffs.index):
#             x, y = coeffs.iloc[i, 0], coeffs.iloc[i, 1]
#             plt.arrow(0, 0, x * 3, y * 3, color='r', alpha=0.7, head_width=0.05)
#             plt.text(x * 3.2, y * 3.2, col, fontsize=9)
#         plt.xlabel('PC1');
#         plt.ylabel('PC2')
#         plt.title('Biplot (scores + loadings arrows)')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
#
#         # 13) Reconstruction error (optional) to see how much info lost
#         X_inv = pca.inverse_transform(scores)  # in scaled space
#         reconstruction_mse = np.mean((df_scaled - X_inv) ** 2)
#         print("Reconstruction MSE (on scaled features):", reconstruction_mse)
#
#         # End: show top summary
#         print("Explained variance by chosen components:", pca.explained_variance_ratio_.sum())
