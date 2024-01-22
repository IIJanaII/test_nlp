import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


import time


# Load Course dataset

df=pd.read_csv("df_mergedV4.csv",sep=',')
filtered_documents=df.copy()

qa = pipeline('question-answering')

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

    elif page == "Chatbot: Question Answering":
        st.header("Try our Assistant chatbot to help you !")
        st.subheader("Hello! How can I help you?")

        
        # Display chat messages from history
        

        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "context_string" not in st.session_state:
            st.session_state.context_string=[]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            

        if prompt  and st.session_state.context_string==[] :
                # Retrieve top documents using TF-IDF
            similarity_scores_with_impact, top_documents = retrieve_top_documents(prompt, k=10)

               

                # Create context string for question answering
            context_string = create_context_string(top_documents)
            st.session_state.context_string = context_string

                # Perform question answering
            
            answer = qa(context=context_string, question=prompt)

            with st.chat_message("assistant"):
                message_placeholder=st.empty()
                full_response=""
                assistant_response=answer['answer']

                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "")
                message_placeholder.markdown(full_response)
             # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})



            
                # Option to refresh or continue with the same context
        elif prompt and st.session_state.context_string!=[]:
                qa = pipeline('question-answering')
                answer = qa(context=st.session_state.context_string, question=prompt)
                print("prompt",prompt)

                with st.chat_message("assistant"):
                    message_placeholder=st.empty()
                    full_response=""
                    assistant_response=answer['answer']

                    for chunk in assistant_response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.text("Continue with the same context. Ask another question.")
            st.session_state.context_string=[]
    

  
    


if __name__ == "__main__":
    main()
