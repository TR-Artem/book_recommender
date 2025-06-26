import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import joblib
from pathlib import Path
import json

class AdvancedBookRecommender:
    def __init__(self):
        self.books, self.ratings = self.load_extended_data()
        self.book_images = self.load_book_images()
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        self._cache = {}
        
        # Определяем стоп-слова здесь, чтобы они были доступны во всех методах
        self.RUSSIAN_STOP_WORDS = [
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
            'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
            'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от'
        ]
        
        self.prepare_models()
    
    def load_book_images(self):
        try:
            with open('data/book_images.json', 'r') as f:
                return json.load(f)
        except:
            return {
                1: "https://m.media-amazon.com/images/I/71kxa1-0mfL._AC_UF1000,1000_QL80_.jpg",
                2: "https://m.media-amazon.com/images/I/81iqZ2HHD-L._AC_UF1000,1000_QL80_.jpg",
                3: "https://m.media-amazon.com/images/I/81o6s0k4XFL._AC_UF1000,1000_QL80_.jpg",
                4: "https://m.media-amazon.com/images/I/71yMgdeZ5FL._AC_UF1000,1000_QL80_.jpg",
                5: "https://m.media-amazon.com/images/I/81XQ1+pkLgL._AC_UF1000,1000_QL80_.jpg",
                6: "https://m.media-amazon.com/images/I/71BSdq6OzDL._AC_UF1000,1000_QL80_.jpg",
                7: "https://m.media-amazon.com/images/I/71Xygne8GrL._AC_UF1000,1000_QL80_.jpg",
                8: "https://m.media-amazon.com/images/I/91li9g2qW8L._AC_UF1000,1000_QL80_.jpg",
                9: "https://m.media-amazon.com/images/I/91RcWcQn61L._AC_UF1000,1000_QL80_.jpg",
                10: "https://m.media-amazon.com/images/I/91ySX+d+6VL._AC_UF1000,1000_QL80_.jpg",
                11: "https://m.media-amazon.com/images/I/71OFqSRFDgL._AC_UF1000,1000_QL80_.jpg",
                12: "https://m.media-amazon.com/images/I/71FxgtFKcQL._AC_UF1000,1000_QL80_.jpg",
                13: "https://m.media-amazon.com/images/I/71jLBXtWJVL._AC_UF1000,1000_QL80_.jpg",
                14: "https://m.media-amazon.com/images/I/91HPG31dTwL._AC_UF1000,1000_QL80_.jpg",
                15: "https://m.media-amazon.com/images/I/91DfS2k6BLL._AC_UF1000,1000_QL80_.jpg"
            }
    
    def load_extended_data(self):
        books = pd.DataFrame({
            'book_id': range(1, 16),
            'title': ['1984', 'Гарри Поттер и философский камень', 'Мастер и Маргарита', 
                     'Преступление и наказание', 'Три товарища', 'Маленький принц',
                     'Атлант расправил плечи', 'Шерлок Холмс', 'Война и мир', 'Горе от ума',
                     '451 градус по Фаренгейту', 'Убить пересмешника', 'Властелин колец',
                     'Над пропастью во ржи', 'Анна Каренина'],
            'author': ['Джордж Оруэлл', 'Дж. К. Роулинг', 'Михаил Булгаков',
                      'Федор Достоевский', 'Эрих Мария Ремарк', 'Антуан де Сент-Экзюпери',
                      'Айн Рэнд', 'Артур Конан Дойл', 'Лев Толстой', 'Александр Грибоедов',
                      'Рэй Брэдбери', 'Харпер Ли', 'Дж. Р. Р. Толкин',
                      'Джером Сэлинджер', 'Лев Толстой'],
            'genre': ['Антиутопия', 'Фэнтези', 'Магический реализм', 'Классика', 'Роман', 
                     'Философская сказка', 'Роман', 'Детектив', 'Исторический роман', 'Комедия',
                     'Антиутопия', 'Роман', 'Фэнтези', 'Роман', 'Классика'],
            'description': [
                'Роман-антиутопия о тоталитарном обществе',
                'Первая книга о юном волшебнике Гарри Поттере',
                'Мистический роман о дьяволе, посещающем Москву',
                'История бывшего студента, совершившего убийство',
                'История дружбы трех товарищей в послевоенной Германии',
                'Философская сказка о маленьком принце с другой планеты',
                'Роман о роли разума в жизни человека и общества',
                'Сборник рассказов о знаменитом детективе',
                'Эпический роман о войне с Наполеоном',
                'Сатирическая комедия о нравах дворянства',
                'Антиутопия о мире, где книги под запретом',
                'История о расовой несправедливости в Америке',
                'Эпическая фэнтези-сага о борьбе за Кольцо Всевластья',
                'История подростка, переживающего экзистенциальный кризис',
                'Трагическая история любви замужней женщины'
            ],
            'popularity': [9.2, 9.5, 9.3, 8.9, 9.0, 9.7, 8.5, 9.1, 8.8, 8.7, 9.0, 9.3, 9.4, 8.9, 9.1]
        })
        
        # Генерация реалистичных рейтингов
        np.random.seed(42)
        num_users = 200
        user_preferences = np.random.normal(loc=0, scale=1, size=(num_users, 3))
        
        book_features = np.array([
            [1,0,0,0,0,0,0,0,0,0,9.2], [0,0,0,0,0,1,0,0,0,0,9.5],
            [0,0,0,0,0,0,1,0,0,0,9.3], [0,1,0,0,0,0,0,0,0,0,8.9],
            [0,0,0,0,1,0,0,0,0,0,9.0], [0,0,0,0,0,0,0,0,1,0,9.7],
            [0,0,0,0,1,0,0,0,0,0,8.5], [0,0,1,0,0,0,0,0,0,0,9.1],
            [0,0,0,1,0,0,0,0,0,0,8.8], [0,0,0,0,0,0,0,1,0,0,8.7],
            [1,0,0,0,0,0,0,0,0,0,9.0], [0,0,0,0,1,0,0,0,0,0,9.3],
            [0,0,0,0,0,1,0,0,0,0,9.4], [0,0,0,0,1,0,0,0,0,0,8.9],
            [0,1,0,0,0,0,0,0,0,0,9.1]
        ])
        
        ratings = []
        for user_id in range(num_users):
            for book_id in range(15):
                if np.random.random() > 0.7:
                    base_rating = np.dot(user_preferences[user_id], book_features[book_id][:3])
                    rating = np.clip((base_rating + np.random.normal(0,0.5) + book_features[book_id][-1]/2 -4)*2+3, 1, 5)
                    ratings.append({
                        'user_id': user_id+1,
                        'book_id': book_id+1,
                        'rating': int(round(rating))
                    })
        
        return books, pd.DataFrame(ratings)

    def prepare_models(self):
        try:
            self.load_models()
            print("Models loaded from cache")
            return
        except:
            print("Training models from scratch...")
        
        self.books['metadata'] = self.books['genre'] + ' ' + self.books['author'] + ' ' + self.books['description']
        
        # TF-IDF
        tfidf = TfidfVectorizer(stop_words=self.RUSSIAN_STOP_WORDS, max_features=5000)
        tfidf_matrix = tfidf.fit_transform(self.books['metadata'])
        self.content_similarity = cosine_similarity(tfidf_matrix)
        
        # Collaborative
        self.user_book_matrix = self.ratings.pivot_table(
            index='user_id', columns='book_id', values='rating', fill_value=0)
        
        svd = TruncatedSVD(n_components=10, random_state=42)
        self.reduced_matrix = svd.fit_transform(self.user_book_matrix.T)
        self.collab_similarity = cosine_similarity(self.reduced_matrix)
        
        # Hybrid
        self.hybrid_model = 0.6*self.content_similarity + 0.4*self.collab_similarity
        
        # KNN
        self.knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn_model.fit(self.reduced_matrix)
        
        # Clustering
        self.cluster_model = KMeans(n_clusters=5, random_state=42)
        self.book_clusters = self.cluster_model.fit_predict(self.reduced_matrix)
        self.books['cluster'] = self.book_clusters
        
        self.save_models()

    def save_models(self):
        joblib.dump({
            'content_similarity': self.content_similarity,
            'reduced_matrix': self.reduced_matrix,
            'collab_similarity': self.collab_similarity,
            'hybrid_model': self.hybrid_model,
            'knn_model': self.knn_model,
            'cluster_model': self.cluster_model,
            'book_clusters': self.book_clusters
        }, self.models_path/"book_models.pkl")

    def load_models(self):
        models = joblib.load(self.models_path/"book_models.pkl")
        self.content_similarity = models['content_similarity']
        self.reduced_matrix = models['reduced_matrix']
        self.collab_similarity = models['collab_similarity']
        self.hybrid_model = models['hybrid_model']
        self.knn_model = models['knn_model']
        self.cluster_model = models['cluster_model']
        self.book_clusters = models['book_clusters']
        self.books['cluster'] = self.book_clusters

    def get_recommendations(self, book_id, method='hybrid', top_n=5):
        cache_key = (book_id, method, top_n)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if method == 'content':
            scores = self.content_similarity[book_id-1]
        elif method == 'collab':
            scores = self.collab_similarity[book_id-1]
        elif method == 'cluster':
            cluster_id = self.books.loc[self.books['book_id']==book_id, 'cluster'].values[0]
            cluster_books = self.books[self.books['cluster']==cluster_id]
            similar_indices = cluster_books['book_id'].apply(lambda x: self.hybrid_model[book_id-1,x-1])
            similar_indices = similar_indices.sort_values(ascending=False).index[:top_n+1]
            result = self.books.iloc[similar_indices]
            result = result[result['book_id']!=book_id].head(top_n)
            self._cache[cache_key] = result
            return result
        else:
            scores = self.hybrid_model[book_id-1]
        
        similar_indices = np.argsort(scores)[-top_n-1:-1][::-1]
        result = self.books.iloc[similar_indices]
        self._cache[cache_key] = result
        return result

    def get_knn_recommendations(self, book_id, top_n=5):
        distances, indices = self.knn_model.kneighbors(
            [self.reduced_matrix[book_id-1]], n_neighbors=top_n+1)
        return self.books.iloc[indices[0][1:]]

    def get_diverse_recommendations(self, book_id, top_n=5):
        content_rec = self.get_recommendations(book_id, 'content', top_n*2)
        collab_rec = self.get_recommendations(book_id, 'collab', top_n*2)
        cluster_rec = self.get_recommendations(book_id, 'cluster', top_n*2)
        all_rec = pd.concat([content_rec, collab_rec, cluster_rec]).drop_duplicates()
        all_rec = all_rec[all_rec['book_id']!=book_id]
        return all_rec.sort_values('popularity', ascending=False).head(top_n)