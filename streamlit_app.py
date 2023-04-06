import streamlit as st
import pinecone
from utils import get_text_embedding

PINECONE_KEY = st.secrets["PINECONE_KEY"]

st.write(
    """
# CLIP Image Search Engine
Hello! This is a demo of a Pinecone-powered image search engine using the CLIP model.
The dataset used is the COCO 2017 dataset, which contains 118,287 images.

To get started, type in a search query and simply press enter.
Try searching for objects more than people, as COCO is a dataset of objects.

Happy searching!
"""
)

pinecone.init(api_key=PINECONE_KEY, environment="us-east4-gcp")  # app.pinecone.io

index_name = "clip-image-search"
index = pinecone.Index(index_name=index_name)

text_query = st.text_input("Search for images", "A cat in the rain")

query_vector = get_text_embedding(text_query)

top_k_samples = index.query(vector=query_vector, top_k=10, include_values=False)

for result in top_k_samples["matches"]:
    st.image(result.id)

st.image(result.id for result in top_k_samples["matches"])
