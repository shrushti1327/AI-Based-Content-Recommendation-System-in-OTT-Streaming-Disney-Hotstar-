import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Disney+ Hotstar AI Recommender", layout="wide")

# -----------------------------
# Disney+ Hotstar Styling
# -----------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom, #0c111b, #101826);
        color: white;
    }
    .stButton>button {
        background-color: #1f80e0;
        color: white;
        border-radius: 20px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Disney+ Hotstar AI Recommendation System")
st.markdown("Premium Personalized Content Experience")

# -----------------------------
# Dataset (No Poster Column)
# -----------------------------
content_data = {
    'title': [
        'Avengers: Endgame',
        'Frozen',
        'MS Dhoni: The Untold Story',
        'The Lion King',
        'IPL 2023 Final',
        'Thor: Ragnarok',
        'Luca',
        'Guardians of the Galaxy'
    ],
    'genre': [
        'Action',
        'Animation',
        'Sports Drama',
        'Animation',
        'Sports',
        'Action',
        'Animation',
        'Action'
    ],
    'description': [
        'Marvel superheroes unite to defeat Thanos.',
        'A magical journey of two royal sisters.',
        'Biopic of Indian cricket captain MS Dhoni.',
        'A young lion prince returns to reclaim his kingdom.',
        'Live Indian Premier League cricket final match.',
        'Thor battles to save Asgard.',
        'A coming-of-age story set in Italy.',
        'A group of space heroes protect the galaxy.'
    ]
}

content_df = pd.DataFrame(content_data)

# -----------------------------
# AI Model
# -----------------------------
tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform(content_df['genre'])
cosine_sim = cosine_similarity(matrix)

def recommend(title):
    idx = content_df[content_df['title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]
    indices = [i[0] for i in scores]
    return content_df.iloc[indices]

# -----------------------------
# UI
# -----------------------------
selected = st.selectbox("🎥 Select Content", content_df['title'])

if st.button("Get Recommendations"):
    recommended_df = recommend(selected)

    st.subheader("✨ Recommended For You")

    for _, row in recommended_df.iterrows():
        st.markdown(f"""
        ### 🎬 {row['title']}
        **Genre:** {row['genre']}  
        {row['description']}
        """)