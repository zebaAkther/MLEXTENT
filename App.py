import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import squarify
import seaborn as sns

# Page title
st.title("Market Basket Analysis with Association Rules")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('/kaggle/input/market-basket-optimisation-4-csv/Market_Basket_Optimisation 4.csv', header=None, names=[f'Item_{i}' for i in range(1, 21)])
    txns = df.fillna('').values.tolist()
    txns = [[item.strip() for item in txn if item != ''] for txn in txns]
    return txns

transactions = load_data()

# One-hot encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_array, columns=te.columns_)

# User inputs for thresholds
min_support = st.sidebar.slider("Minimum support", 0.01, 0.10, 0.04, 0.01)
min_confidence = st.sidebar.slider("Minimum confidence", 0.1, 1.0, 0.3, 0.05)

# Generate frequent itemsets
frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
rules = rules.sort_values(['confidence', 'support', 'lift'], ascending=[False, False, False])

# Show frequent items
st.subheader("Frequent Items")
item_counts = pd.Series([item for sublist in transactions for item in sublist]).value_counts()
st.dataframe(item_counts.reset_index().rename(columns={'index': 'Item', 0: 'Count'}).head(20))

# Show association rules table
st.subheader("Association Rules")
st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Show word cloud of items
st.subheader("Item Word Cloud")
all_items = ' '.join([item for sublist in transactions for item in sublist])
wordcloud = WordCloud(width=400, height=200, background_color='white').generate(all_items)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Treemap of top frequent items
st.subheader("Top 30 Frequent Items Tree Map")
top30 = item_counts.head(30)
plt.figure(figsize=(12, 6))
labels = [f"{item}\n({count})" for item, count in zip(top30.index, top30.values)]
colors = sns.color_palette('Spectral', len(top30))
squarify.plot(sizes=top30.values, label=labels, color=colors, alpha=0.8)
plt.axis('off')
st.pyplot(plt)

# Optional: Scatterplot for support vs confidence visualization of rules
st.subheader("Support vs Confidence of Association Rules")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rules, x='support', y='confidence', size='confidence', sizes=(20, 200), legend=False)
plt.xlabel('Support')
plt.ylabel('Confidence')
st.pyplot(plt)
