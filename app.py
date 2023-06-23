import pickle
import streamlit as st
import numpy as np

st.header('Book Recommendation System using machine learning (collaborative filtring)')
model = pickle.load(open('model.pkl','rb'))
book_names = pickle.load(open('book_names.pkl','rb'))
final_rating = pickle.load(open('final_rating.pkl','rb'))
book_pivot = pickle.load(open('book_pivot.pkl','rb'))

def books_url(suggestion):
  books_names = []
  ids_index = []
  poster_url = []

  if len(suggestion != 0):
    for book_id in suggestion:
      books_names.append(book_pivot.index[book_id])
    
    for name in books_names[0]:
      ids = np.where(final_rating['title'] == name)[0][0]
      ids_index.append(ids)

    for id in ids_index:
      url = final_rating.iloc[id]['img_url']
      poster_url.append(url)

  return poster_url



def recommand_books(book_name):
  books_names = []
  book_index = np.where(book_pivot.index == book_name)[0][0]
  distance, suggestion = model.kneighbors(book_pivot.iloc[book_index,:].values.reshape(1,-1), n_neighbors=6 )

  posters_url = books_url(suggestion)
  for i in range(len(suggestion)):
    books = book_pivot.index[suggestion[i]]
    for j in books:
       books_names.append(j)

  return books_names, posters_url

selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books,poster_url = recommand_books(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_books[1])
        st.image(poster_url[1])
    with col2:
        st.text(recommended_books[2])
        st.image(poster_url[2])

    with col3:
        st.text(recommended_books[3])
        st.image(poster_url[3])
    with col4:
        st.text(recommended_books[4])
        st.image(poster_url[4])
    with col5:
        st.text(recommended_books[5])
        st.image(poster_url[5])

