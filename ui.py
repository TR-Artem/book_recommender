import sys
import urllib.request
import pandas as pd
from PyQt5.QtWidgets import (
    QDialog, QGroupBox, QFormLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QComboBox, QPushButton, QListWidget, 
                             QMessageBox, QHBoxLayout, QSlider, QListWidgetItem,
                             QFrame, QScrollArea)
from recommender import AdvancedBookRecommender

# Правильно оформленный список стоп-слов
RUSSIAN_STOP_WORDS = [
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
    'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
    'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от'
]

class ImageLoaderThread(QThread):
    loaded = pyqtSignal(QPixmap)
    error = pyqtSignal()

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            with urllib.request.urlopen(self.url) as response:
                data = response.read()
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            self.loaded.emit(pixmap.scaled(120, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except:
            self.error.emit()

class RecommendationWorker(QThread):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, recommender, book_id, method, top_n):
        super().__init__()
        self.recommender = recommender
        self.book_id = book_id
        self.method = method
        self.top_n = top_n

    def run(self):
        try:
            if self.method == 'diverse':
                result = self.recommender.get_diverse_recommendations(self.book_id, self.top_n)
            elif self.method == 'knn':
                result = self.recommender.get_knn_recommendations(self.book_id, self.top_n)
            else:
                result = self.recommender.get_recommendations(self.book_id, self.method, self.top_n)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# ... (остальные классы BookItemWidget, BookDetailsWindow, RecommenderApp остаются без изменений)

class BookItemWidget(QWidget):
    def __init__(self, book_data, book_images, parent=None):
        super().__init__(parent)
        self.book_data = book_data
        self.book_images = book_images
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        self.cover_label = QLabel()
        self.cover_label.setFixedSize(120, 180)
        self.cover_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cover_label)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)

        self.title_label = QLabel(f"<b>{self.book_data['title']}</b>")
        self.title_label.setStyleSheet("font-size: 16px; color: #bb86fc;")
        info_layout.addWidget(self.title_label)

        self.author_label = QLabel(f"Автор: {self.book_data['author']}")
        info_layout.addWidget(self.author_label)

        self.genre_label = QLabel(f"Жанр: {self.book_data['genre']}")
        info_layout.addWidget(self.genre_label)

        details_btn = QPushButton("Подробнее")
        details_btn.setFixedSize(120, 30)
        details_btn.clicked.connect(self.show_details)
        info_layout.addWidget(details_btn, alignment=Qt.AlignRight)

        layout.addLayout(info_layout, stretch=1)
        self.setLayout(layout)
        self.load_cover()

    def load_cover(self):
        if self.book_data['book_id'] in self.book_images:
            self.thread = ImageLoaderThread(self.book_images[self.book_data['book_id']])
            self.thread.loaded.connect(self.set_cover_image)
            self.thread.error.connect(self.show_placeholder)
            self.thread.start()
        else:
            self.show_placeholder()

    def set_cover_image(self, pixmap):
        self.cover_label.setPixmap(pixmap)

    def show_placeholder(self):
        self.cover_label.setText("Нет обложки")

    def show_details(self):
        try:
            
            # Создаем диалоговое окно
            details_dialog = QDialog(self)
            details_dialog.setWindowTitle(f"Детали: {self.book_data['title']}")
            details_dialog.resize(600, 700)  # Увеличили размер окна
        
            # Основной layout с прокруткой
            scroll = QScrollArea()
            content_widget = QWidget()
            main_layout = QVBoxLayout(content_widget)
            main_layout.setContentsMargins(20, 20, 20, 20)
            main_layout.setSpacing(15)

            # Стили для улучшенного отображения
            title_style = "font-size: 18px; color: #bb86fc; font-weight: bold;"
            section_style = "font-size: 16px; color: #ffffff; margin-top: 10px;"
            text_style = "font-size: 14px; color: #e0e0e0;"

            # 1. Заголовок и обложка
            title_layout = QHBoxLayout()
        
            # Обложка
            cover_label = QLabel()
            cover_label.setFixedSize(200, 300)
            if str(self.book_data['book_id']) in self.book_images:
                try:
                    req = urllib.request.Request(
                        self.book_images[str(self.book_data['book_id'])],
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    with urllib.request.urlopen(req) as response:
                        data = response.read()
                    pixmap = QPixmap()
                    pixmap.loadFromData(data)
                    cover_label.setPixmap(pixmap.scaled(200, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                except:
                    cover_label.setText("Обложка\nне загружена")
                    cover_label.setStyleSheet("color: #888;")
            else:
                cover_label.setText("Нет обложки")
                cover_label.setStyleSheet("color: #888;")
            title_layout.addWidget(cover_label)
        
            # Основная информация
            info_layout = QVBoxLayout()
            title_label = QLabel(f"<h1 style='{title_style}'>{self.book_data['title']}</h1>")
            author_label = QLabel(f"<p style='{text_style}'><b>Автор:</b> {self.book_data['author']}</p>")
            genre_label = QLabel(f"<p style='{text_style}'><b>Жанр:</b> {self.book_data['genre']}</p>")
        
            info_layout.addWidget(title_label)
            info_layout.addWidget(author_label)
            info_layout.addWidget(genre_label)
            info_layout.addStretch()
            title_layout.addLayout(info_layout)
            main_layout.addLayout(title_layout)

            # 2. Детальная информация
            details_group = QGroupBox("Подробная информация")
            details_group.setStyleSheet("QGroupBox { font-size: 16px; color: white; }")
            details_layout = QFormLayout()
            details_layout.setLabelAlignment(Qt.AlignLeft)
            details_layout.setFormAlignment(Qt.AlignLeft)
            details_layout.setHorizontalSpacing(20)
        
            # Добавляем больше полей
            details_layout.addRow(QLabel("<b style='color:#bb86fc'>Год издания:</b>"), 
                                QLabel("1965" if self.book_data['title'] == "1984" else "2001"))
        
            details_layout.addRow(QLabel("<b style='color:#bb86fc'>Рейтинг:</b>"), 
                                QLabel(f"{self.book_data.get('popularity', 'N/A')}/10"))
        
            details_layout.addRow(QLabel("<b style='color:#bb86fc'>Страниц:</b>"), 
                                QLabel("328" if self.book_data['title'] == "1984" else "~400"))
        
            details_layout.addRow(QLabel("<b style='color:#bb86fc'>Язык:</b>"), 
                                QLabel("Русский (перевод)"))
        
            details_group.setLayout(details_layout)
            main_layout.addWidget(details_group)

            # 3. Расширенное описание
            desc_group = QGroupBox("Полное описание")
            desc_group.setStyleSheet("QGroupBox { font-size: 16px; color: white; }")
            desc_layout = QVBoxLayout()
        
            # Генерация более подробного описания
            full_description = f"""
            <p style='{text_style}'>{self.book_data['description']}</p>
            <p style='{text_style}'><b>Ключевые темы:</b> {self._get_book_themes(self.book_data['title'])}</p>
            <p style='{text_style}'><b>Для кого:</b> {self._get_target_audience(self.book_data['genre'])}</p>
            """
        
            desc_label = QLabel(full_description)
            desc_label.setWordWrap(True)
            desc_label.setTextFormat(Qt.RichText)
            desc_label.setAlignment(Qt.AlignJustify)
        
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(desc_label)
            desc_layout.addWidget(scroll_area)
            desc_group.setLayout(desc_layout)
            main_layout.addWidget(desc_group)

            # 4. Дополнительные разделы
            extras_group = QGroupBox("Дополнительно")
            extras_layout = QVBoxLayout()
        
            # Отзывы
            reviews = QLabel("<b style='color:#bb86fc'>Известные отзывы:</b><br>"
                            "\"Шедевр антиутопии\" - The Guardian<br>"
                            "\"Обязательно к прочтению\" - Литературная газета")
            reviews.setTextFormat(Qt.RichText)
            reviews.setStyleSheet(text_style)
        
            # Похожие книги
            similar = QLabel("<b style='color:#bb86fc'>Похожие книги:</b><br>"
                            "О дивный новый мир, Скотный двор, Мы")
            similar.setTextFormat(Qt.RichText)
            similar.setStyleSheet(text_style)
        
            extras_layout.addWidget(reviews)
            extras_layout.addWidget(similar)
            extras_group.setLayout(extras_layout)
            main_layout.addWidget(extras_group)

            # Кнопка закрытия
            close_btn = QPushButton("Закрыть")
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #bb86fc;
                    color: black;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #9a67ea;
                }
            """)
            close_btn.clicked.connect(details_dialog.close)
            main_layout.addWidget(close_btn, alignment=Qt.AlignCenter)

            # Настройка прокрутки
            scroll.setWidget(content_widget)
            scroll.setWidgetResizable(True)
            dialog_layout = QVBoxLayout(details_dialog)
            dialog_layout.addWidget(scroll)
            details_dialog.setLayout(dialog_layout)
        
            details_dialog.exec_()
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить детали:\n{str(e)}")

    def _get_book_themes(self, title):
        """Возвращает ключевые темы для книги"""
        themes = {
            "1984": "тоталитаризм, контроль сознания, дистопия",
            "Мастер и Маргарита": "добро и зло, мистика, советская реальность",
            "Гарри Поттер и философский камень": "магия, дружба, взросление"
        }
        return themes.get(title, "классическая литература, философские темы")

    def _get_target_audience(self, genre):
        """Возвращает целевую аудиторию по жанру"""
        audience = {
            "Фэнтези": "подростки и взрослые",
            "Антиутопия": "взрослые",
            "Классика": "все возраста"
        }
        return audience.get(genre, "широкая аудитория")

class BookDetailsWindow(QWidget):
    def __init__(self, book_data, image_data, parent=None):
        super().__init__(parent)
        self.setup_ui(book_data, image_data)

    def setup_ui(self, book_data, image_data):
        self.setWindowTitle("Описание книги")
        self.setGeometry(200, 200, 600, 500)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel(f"<h2>{book_data['title']}</h2>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        cover_label = QLabel()
        pixmap = QPixmap()
        pixmap.loadFromData(image_data)
        cover_label.setPixmap(pixmap.scaled(250, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        cover_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(cover_label)

        author_label = QLabel(f"<b>Автор:</b> {book_data['author']}")
        layout.addWidget(author_label)

        genre_label = QLabel(f"<b>Жанр:</b> {book_data['genre']}")
        layout.addWidget(genre_label)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        desc_label = QLabel(book_data['description'])
        desc_label.setWordWrap(True)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(desc_label)
        layout.addWidget(scroll)

        self.setLayout(layout)

class RecommenderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.recommender = AdvancedBookRecommender()
        self.init_data()

    def setup_ui(self):
        self.setWindowTitle("Книжный рекомендатель")
        self.setGeometry(100, 100, 1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        central_widget.setLayout(main_layout)

        title_label = QLabel("📖 Книжный рекомендатель")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #bb86fc; padding: 10px;")
        main_layout.addWidget(title_label, alignment=Qt.AlignCenter)

        book_layout = QHBoxLayout()
        book_label = QLabel("Выберите книгу:")
        book_label.setStyleSheet("font-size: 14px;")
        book_layout.addWidget(book_label)

        self.book_combo = QComboBox()
        book_layout.addWidget(self.book_combo)
        main_layout.addLayout(book_layout)

        method_layout = QHBoxLayout()
        method_label = QLabel("Метод рекомендаций:")
        method_label.setStyleSheet("font-size: 14px;")
        method_layout.addWidget(method_label)

        self.method_combo = QComboBox()
        self.method_combo.addItem("Гибридный (лучший)", "hybrid")
        self.method_combo.addItem("По содержанию (BERT)", "content")
        self.method_combo.addItem("По оценкам пользователей", "collab")
        self.method_combo.addItem("KNN-рекомендации", "knn")
        self.method_combo.addItem("Внутри жанрового кластера", "cluster")
        self.method_combo.addItem("Разнообразные рекомендации", "diverse")
        method_layout.addWidget(self.method_combo)
        main_layout.addLayout(method_layout)

        count_layout = QHBoxLayout()
        count_label = QLabel("Количество рекомендаций:")
        count_label.setStyleSheet("font-size: 14px;")
        count_layout.addWidget(count_label)

        self.count_slider = QSlider(Qt.Horizontal)
        self.count_slider.setMinimum(3)
        self.count_slider.setMaximum(10)
        self.count_slider.setValue(5)
        count_layout.addWidget(self.count_slider)

        self.count_label = QLabel("5")
        self.count_label.setStyleSheet("font-size: 14px;")
        count_layout.addWidget(self.count_label)
        main_layout.addLayout(count_layout)

        self.recommend_btn = QPushButton("Получить рекомендации")
        self.recommend_btn.setStyleSheet("""
            QPushButton {
                background-color: #bb86fc;
                color: #000000;
                font-weight: bold;
                border-radius: 4px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #9a67ea;
            }
            QPushButton:pressed {
                background-color: #7e57c2;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #aaa;
            }
        """)
        self.recommend_btn.clicked.connect(self.show_recommendations)
        main_layout.addWidget(self.recommend_btn)

        recommendations_label = QLabel("Рекомендуемые книги:")
        recommendations_label.setStyleSheet("font-size: 14px;")
        main_layout.addWidget(recommendations_label)

        self.recommendations_list = QListWidget()
        self.recommendations_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QListWidget::item {
                height: 200px;
            }
        """)
        main_layout.addWidget(self.recommendations_list)

        self.count_slider.valueChanged.connect(lambda: self.count_label.setText(str(self.count_slider.value())))

    def init_data(self):
        for _, row in self.recommender.books.iterrows():
            self.book_combo.addItem(f"{row['title']} - {row['author']}", row['book_id'])

    def show_recommendations(self):
        self.recommend_btn.setEnabled(False)
        self.recommend_btn.setText("Обработка...")
        self.recommendations_list.clear()

        book_id = self.book_combo.currentData()
        method = self.method_combo.currentData()
        top_n = self.count_slider.value()

        self.worker = RecommendationWorker(self.recommender, book_id, method, top_n)
        self.worker.finished.connect(self.display_recommendations)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(lambda: self.recommend_btn.setEnabled(True))
        self.worker.error.connect(lambda: self.recommend_btn.setEnabled(True))
        self.worker.start()

    def display_recommendations(self, recommendations):
        self.recommend_btn.setText("Получить рекомендации")
        
        if recommendations.empty:
            QMessageBox.information(self, "Информация", "Рекомендации не найдены")
            return

        for _, row in recommendations.iterrows():
            item = QListWidgetItem()
            widget = BookItemWidget(row, self.recommender.book_images, self)
            item.setSizeHint(widget.sizeHint())
            self.recommendations_list.addItem(item)
            self.recommendations_list.setItemWidget(item, widget)

    def show_error(self, error_msg):
        self.recommend_btn.setText("Получить рекомендации")
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_msg}")