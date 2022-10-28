from typing import Sequence
from aleph_alpha_client import (
    AlephAlphaClient,
    Prompt,
    AlephAlphaModel,
    CompletionRequest,
    SemanticEmbeddingRequest,
    SemanticRepresentation,
)
from scipy.spatial import distance

import os
import math

import streamlit as st


search_model = AlephAlphaModel.from_model_name(
    model_name="luminous-base", token=os.getenv("API_TOKEN")
)

model = AlephAlphaModel.from_model_name(
    model_name="luminous-extended", token=os.getenv("API_TOKEN")
)


# function for symmetric embeddings
def embed_symmetric(text: str):
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text(text), representation=SemanticRepresentation.Symmetric
    )
    result = search_model.semantic_embed(request)
    return result.embedding


# function for asymmetric embeddings of Queries
def embed_query(text: str):
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text(text), representation=SemanticRepresentation.Query
    )
    result = search_model.semantic_embed(request)
    return result.embedding


# function for asymmetric embeddings of Documents
def embed_document(text: str):
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text(text), representation=SemanticRepresentation.Document
    )
    result = search_model.semantic_embed(request)
    return result.embedding


# function to calculate similarity
def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


# function to complete a simple text
def complete(text: str):
    request = CompletionRequest(
        prompt=Prompt.from_text(text),
        maximum_tokens=32,
        temperature=0.2,
        stop_sequences=["\n"],
    )
    result = model.complete(request)
    return result.completions[0].completion


# A minimal streamlit interface for accessing the model

st.title("Luminous Streamlit Demo")

st.markdown("#### This is a minimal demo of serving the Luminous models via Streamlit")

# input text that will be completed
input_text = st.text_input(
    "Enter a text to complete",
    "Luminous are a family of Large Language Models. Luminous Models are capable of",
)

# button to complete the text
if st.button("Complete"):
    completion = complete(input_text)
    st.markdown(f"#### Completion:")
    st.write(completion)
