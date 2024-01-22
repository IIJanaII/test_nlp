import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel




# Load Course dataset

df=pd.read_csv("df_mergedV4.csv",sep=',')
filtered_documents=df.copy()

@st.cache_data(persist=True)
def get_tfidf_vectorizer(description_trad_clean):
    # Create a TF-IDF vectorizer with specific settings
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    return tfidf_vectorizer.fit(description_trad_clean)

# Retrieve the TF-IDF vectorizer, cached for efficiency
tfidf_vectorizer = get_tfidf_vectorizer(filtered_documents['description_trad_clean'])


@st.cache_data(persist=True)
def retrieve_top_documents(query_summary, k=10):
    # Transform the query summary into a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query_summary])
    
    # Calculate cosine similarity between the query and all articles
    similarity_scores = linear_kernel(query_vector, tfidf_vectorizer.transform(filtered_documents['description_trad_clean']))
    
    # Add the impact of the average score to the similarity scores
    similarity_scores_with_impact = similarity_scores + filtered_documents['average_score'].values.reshape(1, -1) * 0.1  # Adjust the impact factor as needed
    
    # Sort document indices by the modified similarity score in descending order
    document_indices = similarity_scores_with_impact[0].argsort()[:-k-1:-1]
    
    # Retrieve the top-k documents based on their indices
    top_documents = filtered_documents.iloc[document_indices] 
    
    return similarity_scores_with_impact, top_documents

# Function to create context string for top documents
def create_context_string(top_documents):
    unique_names = top_documents["name"].unique()

    contexts = []
    for name in unique_names:
        subset_df = df[df["name"] == name]
        avg_score = subset_df["average_score"].mean()
        description = subset_df['description_trad_clean'].iloc[0]
        phone_number = subset_df["phone_number"].iloc[0]  # Assuming it's the same for each entry
        context_string = f"The Name of the company is: {name},  {name}'s Description is: {description} {name}'s Average Score is: {avg_score} and {name}'s Phone Number is: {phone_number}"
        contexts.append(context_string)
    
    return '\n'.join(contexts)
import string

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Add any other preprocessing steps as needed

    return text




# Streamlit App
def main():
    st.title("Home services Application")

     # Sidebar with options
    page_options = ["Home", "Prediction","Service Retrieval","Chatbot: Question Answering","Summary","Explanation"]
    page = st.sidebar.selectbox("Choose a page", page_options)

    

    if page == "Home":
        
        st.image("https://global.hitachi-solutions.com/wp-content/uploads/2022/01/Webinar-NLP-In-RCG.png", width=750, caption="Welcome to the NLP App")
        
        st.markdown(
        """
        # Welcome to our NLP application on home services.
        In this application, you will find different use case of NLP techniques on a scrapped data base.
        Use The sidebar to navigate to different pages.

        Have Fun !

        Janany & Mathilde
        """
    )
    





    elif page == "Service Retrieval":
        st.header("Service Retrieval Page")
        query_summary = st.text_area("✏️ Enter your request :")

        if st.button("Retrieve Services"):
            if query_summary:
                # Retrieve top documents using TF-IDF
                similarity_scores_with_impact, top_documents = retrieve_top_documents(query_summary, k=10)

                # Display the query summary
                st.subheader("Your are looking for:")
                st.write(query_summary)

                # Display the top 10 results
                st.subheader("Top 10 Services:")
                for i, (index, row) in enumerate(top_documents.iterrows(), 1):
                    st.write(f"{i}. **{row['name']}**")
                    st.write(f"   Average Score: {row['average_score']:.2f} ⭐️")

                    # Calculate impact based on average score (customize the impact calculation as needed)
                    impact = row['average_score'] * 0.1  # Adjust the multiplication factor as needed
                   

                    # Display the modified similarity score with impact
                    st.write(f"   Similarity Score with Impact: {similarity_scores_with_impact[0][index]:.4f} 🚀")

                    # Expander for additional details
                    with st.expander("Show Details"):
                        st.write(f"   🎯 Description: {row['description_trad_clean']}")
                        st.write(f"   🔗 Link: {row['link']}")
                        st.write(f"   📍 Location: {row['location']}")
                        st.write(f"   📧 Email: {row['email']}")
                        st.write(f"   📞 Phone Number: {row['phone_number']}")

                    st.write("   ---")
            else:
                st.warning("Please enter a query summary before retrieving services.")
    


if __name__ == "__main__":
    main()
