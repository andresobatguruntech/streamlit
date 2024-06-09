import streamlit as st
from sentence_transformers import SentenceTransformer
from umap import UMAP
import pandas as pd
import numpy as np
import string
import spacy
import os
from spacy.lang.id.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial import distance

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

nlp = spacy.blank("id")

def home_page():
    st.title("Home Page")
    st.header("Selamat datang di sistem Topic Modeling")
    st.write("Topic Modeling dilakukan menggunakan BERTopic untuk menganalisis topik-topik dalam artikel berita online terkait Pemilihan Umum (Pemilu) Indonesia 2024. Data yang digunakan merupakan meta deskripsi artikel berita pada sub kanal pemilu di Detik.com mulai dari tanggal 1 September 2023 hingga 14 Februari 2024")

def proses_page():
    st.title("BERTopic Page")        

    if st.button("Prosess"):      
        
        st.write("Proses Membutuhkan waktu, mohon tunggu hingga seluruh proses selesai") 

        df = pd.read_csv("https://raw.githubusercontent.com/andresobatguruntech/streamlit/main/detik-fiks.csv")
        df_subset = df.head(15020)        
        df_subset.to_csv("dataset.csv", index=False)
        st.write("Tahap Prepocessing : Start, Please Wait") 

        
        df = pd.read_csv('dataset.csv')
        df['description_lower_case'] = df['description'].str.lower()
        df.to_csv('dataset.csv', index=False)
        st.write("Tahap Case Folding : Finish") 

        
        df = pd.read_csv('dataset.csv')
        def remove_punctuation(text):
            return text.translate(str.maketrans('', '', string.punctuation))
        
        df['description_clean'] = df['description_lower_case'].apply(remove_punctuation)
        df.to_csv('dataset.csv', index=False)   
        st.write("Tahap Cleaning : Finish")     

        df = pd.read_csv("dataset.csv")
        nlp = spacy.blank("id")
        def remove_stopwords(text):
            doc = nlp(text)
            filtered_text = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
            return " ".join(filtered_text)
        df['description_stopword_removal'] = df['description_clean'].apply(remove_stopwords)
        df.to_csv("dataset.csv", index=False)
        st.write("Tahap Stopword Removal : Finish")  

        train = pd.read_csv('dataset.csv')
        docs = train['description_clean'].to_list()
        st.write("Hasil Preprocessing")
        train[['description', 'description_lower_case', 'description_clean', 'description_stopword_removal']]
        st.write("Tahap Prepocessing : Finish") 

        st.write("Tahap Evaluasi Clustering : Start, Please Wait")
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        embendding = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        docs_ = embendding.encode(docs)
        docs_ = umap_model.fit_transform(docs_)

        st.write("Elbow Method dengan KMeans") 
        model = KMeans(random_state=0)
        visualizer = KElbowVisualizer(model, k=(2,10), metric='distortion')
        visualizer.fit(docs_)
        visualizer.finalize()
        plt.savefig('elbow_plot.png')
        plt.clf()
        st.image('elbow_plot.png')

        st.write("Silhouette Visualizer dengan KMeans") 
        kmeans = KMeans(n_clusters=5, random_state=0)
        visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
        visualizer.fit(docs_)
        visualizer.finalize()
        plt.savefig('silhouette_plot.png')
        plt.clf()
        st.image('silhouette_plot.png')

        st.write("Cluster Visualizer dengan KMeans") 
        kmeans.fit(docs_)
        # Plotting the clusters
        plt.figure(figsize=(8, 6))
        plt.scatter(docs_[:, 0], docs_[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.8)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
        plt.title('Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig('cluster_visualizer.png')
        plt.clf()
        st.image('cluster_visualizer.png')

        silhouette_avg = silhouette_score(docs_, kmeans.labels_)
        st.write(f'Silhouette Score: {silhouette_avg}')

        # Evaluate clustering using Sum of Squared Errors (SSE)
        sse = np.sum((docs_ - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
        st.write(f'SSE: {sse}')

        # Evaluate clustering using Davies-Bouldin Index
        db_index = davies_bouldin_score(docs_, kmeans.labels_)
        st.write(f'Davies-Bouldin Index: {db_index}')

        # Evaluate clustering using Calinski-Harabasz Index (Chi)
        calinski_harabasz_index = calinski_harabasz_score(docs_, kmeans.labels_)
        st.write(f'Calinski-Harabasz Index: {calinski_harabasz_index}')

        # Dunn Index
        def dunn_index(X, labels):
            min_inter_cluster_distance = np.inf
            max_intra_cluster_diameter = -np.inf
            for i in np.unique(labels):
                cluster_points = X[labels == i]
                max_intra_cluster_diameter = max(max_intra_cluster_diameter, np.max(distance.pdist(cluster_points)))
                for j in np.unique(labels):
                    if i != j:
                        other_cluster_points = X[labels == j]
                        min_inter_cluster_distance = min(min_inter_cluster_distance, np.min(distance.cdist(cluster_points, other_cluster_points)))
            dunn = min_inter_cluster_distance / max_intra_cluster_diameter
            return dunn

        dunn = dunn_index(docs_, kmeans.labels_)
        st.write(f'Dunn Index: {dunn}')
        st.write("Tahap Evaluasi Clustering : Finish")

        st.write("Tahap BERTopic : Start, Please Wait")         
        topic_model = BERTopic(embedding_model=None, umap_model=None, hdbscan_model=kmeans)
        topics, _ = topic_model.fit_transform(docs, docs_)
        topics_info = topic_model.get_topics()

        def calculate_coherence(topics_info, docs):
            coherence_values = []
            for topic_id in topics_info:
                topic_words = topics_info[topic_id]
                topic_string = ' '.join([str(word) for word in topic_words])
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(docs)
                topic_vectors = vectorizer.transform([topic_string])
                similarity_matrix = cosine_similarity(X, topic_vectors)
                coherence = similarity_matrix.mean()
                coherence_values.append(coherence)
            return coherence_values
        coherence_values = calculate_coherence(topics_info, docs)

        topic_info = topic_model.get_topic_info()
        df = pd.DataFrame(topic_info)
        df.to_csv("topic_info_all.csv",index=False)

        data = {"ID Topic": [f"{i}" for i in range(len(coherence_values))],
                "Koherensi": coherence_values
                }
        df = pd.DataFrame(data)
        df.to_csv('nilai_topic_koherensi.csv', index=False)

        topic_info_all =  pd.read_csv("topic_info_all.csv")
        nilai_topic_koherensi = pd.read_csv("nilai_topic_koherensi.csv")

        merged_df = pd.merge(topic_info_all, nilai_topic_koherensi, left_on ='Topic', right_on='ID Topic',how='left')

        merged_df.drop(columns=['ID Topic'], inplace=True)
        merged_df.to_csv('topic_info_all_koherensi.csv', index=False)

        if os.path.exists("nilai_topic_koherensi.csv") :
            os.remove("nilai_topic_koherensi.csv")

        topic_info_all_koherensi =  pd.read_csv("topic_info_all_koherensi.csv")
        topic_info_all_koherensi[:10]
        st.write("Tahap BERTopic : Finish") 

        st.write("Informasi Topic :")  
        topic_model.get_topic_info()

        topic_info = topic_model.get_topic_info()
        df = pd.DataFrame(topic_info)
        df.to_csv("topic_info_all.csv",index=False)
        st.write("Informasi Topic disimpan di topic_info_all.csv")  

        topic_model.get_topic_info()[:10]
        topic_model.get_topic(0)

        st.write("Informasi Visual :")  
        st.write("Informasi Visual Topic:")  
        topic_model.visualize_topics()
        visual1 = topic_model.visualize_topics()             
        st.write(visual1)


        st.write("Informasi Visual Hirarki:")  
        topic_model.visualize_hierarchy()
        visual2 = topic_model.visualize_hierarchy()            
        st.write(visual2)

        st.write("Informasi Visual BarChart:")  
        topic_model.visualize_barchart()
        visual3 = topic_model.visualize_barchart()      
        st.write(visual3)

        st.write("Informasi Visual Heatmap:")  
        topic_model.visualize_heatmap()
        visual4 = topic_model.visualize_heatmap()
        st.write(visual4)

        st.write("Informasi Visual Term Rank:")  
        topic_model.visualize_term_rank()
        visual5 = topic_model.visualize_term_rank()
        st.write(visual5)

        st.write("Word Cloud")        
        df = pd.read_csv("topic_info_all_koherensi.csv")
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(train['description_stopword_removal'])

        word_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out() ).mean()
        top_words = word_scores.nlargest(30)
        top_words

        all_representations = ' '.join(top_words.index)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_representations)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud dari Representasi Topik')
        plt.savefig('word_cloud.png')
        plt.clf()
        st.image('word_cloud.png')      
        st.write("Seluruh Proses selesai")
        st.success("Proses Finish")


        st.write("Menampilkan Informasi Kelompok")  
        topic_model.get_document_info(docs)
        document_info = topic_model.get_document_info(docs)
        df = pd.DataFrame(document_info)
        df.to_csv('resultTopic.csv', index=False)

        df = pd.read_csv('resultTopic.csv')
        df
        st.write("Informasi Kelompok success disimpan di resultTopic.csv")  


        st.write("Seluruh Prosess selesai") 
        st.success("Prosess Finish")

def informasi_artikel():
    if st.button("Tampilkan"):
        try:
            st.title("Informasi Artikel")       
            df = pd.read_csv('resultTopic.csv')
            df.iloc[:, 0:6]
    
            st.write("Finish") 
            st.success("Prosess Finish")
        except:
            st.warning('Harap lakukan proses pemodelan topik terlebih dahulu di menu BERTopic', icon="⚠️")

pages = {
    "Home"   : home_page,    
    "BERTopic": proses_page,
    "Informasi Artikel": informasi_artikel,
}
selection = st.sidebar.radio("Menu", list(pages.keys()))
pages[selection]()
