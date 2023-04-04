import streamlit as st
import pandas as pd
import pinecone
from config import PINECONE_KEY
from utils import get_single_text_embedding

st.write(
    """
# My first app
Hello *world!*
"""
)

pinecone.init(api_key=PINECONE_KEY, environment="us-east4-gcp")  # app.pinecone.io

my_index_name = "clip-image-search"
my_index = pinecone.Index(index_name=my_index_name)

text_query = st.text_input(
    "Search for images", "actor arrives for the premiere of the film"
)

query_embedding = get_single_text_embedding(text_query).tolist()

query = my_index.query(query_embedding, top_k=5, include_metadata=True)


for result in query["matches"]:
    st.image(
        result["metadata"]["image"],
        width=200,  # Manually Adjust the width of the image as per requirement
    )
