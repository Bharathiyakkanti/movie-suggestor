import seaborn as sns
import matplotlib.pyplot as plt
from recommender import user_similarity_df

def plot_similarity_heatmap():
    plt.figure(figsize=(10, 6))
    sns.heatmap(user_similarity_df.iloc[:10, :10], cmap='coolwarm')
    plt.title("User Similarity Heatmap")
    plt.show()
