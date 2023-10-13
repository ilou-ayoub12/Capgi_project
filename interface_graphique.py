#pip install -U pyinstaller
#pyinstaller --onefile --noconsole final_Interface.py

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QLineEdit, QMessageBox, QFileDialog, QDialog, QInputDialog, QRadioButton, QButtonGroup, QFormLayout
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt
from language_tool_python import LanguageTool
import sys, openpyxl, Levenshtein, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def calculer_similarite(ph1, ph2):
    distance = Levenshtein.distance(ph1, ph2)
    SM = 1 - distance / max(len(ph1), len(ph2))
    return SM
def prediction(ph):
    df = pd.read_excel('DATA_TOTAL_Diminue_%P_pret_1.xlsx')
    sentences = df['Phrase'].tolist()
    labels = df['Label'].tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    vocab_size = len(tokenizer.word_index) + 1
    max_sequence_length = max(len(seq) for seq in sequences)
    #print(max_sequence_length)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    labels = np.array(labels)
    np.random.seed(42)
    tf.random.set_seed(42)
    model = tf.keras.models.load_model('model_interface.h5')
    new_sequence = tokenizer.texts_to_sequences([ph])
    new_sequence_padded = pad_sequences(new_sequence, maxlen=max_sequence_length)
    predicted_percentage = model.predict(new_sequence_padded)[0][0]
    return predicted_percentage

class SentenceValidationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    def init_ui(self):
        self.setWindowTitle('Sentence Validation App')
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("""
                            QMainWindow {
                                background-color: #f0f0f0;
                            }
                            QLabel#label_1 {
                                font-size: 16px;
                                font-weight: bold;
                                color: #333333;
                                margin-top: 14px;
                                margin-bottom: 14px;
                            }
                            QLabel#etat {
                                font-size: 16px;
                                font-weight: bold;
                                color: #333333;
                                margin-top: 10px;
                                margin-bottom: 10px;
                            }
                            QPushButton#btn_process_data {
                                background-color: #007BFF;
                                color: white;
                                border: none;
                                border-radius: 4px;
                                padding: 10px 20px;
                                font-size: 16px;
                            }
                            QLabel#titleLabel {
                                font-size: 24px;
                                color: #336699;
                                padding: 10px;
                                background-color: #f0f0f0;
                                border-top: 2px solid #336699;
                            }
                        """)
        self.title_label = QLabel("Sentence Validation Application", self)
        self.title_label.setObjectName("titleLabel")
        self.title_label.setTextInteractionFlags(self.title_label.textInteractionFlags() | QtCore.Qt.TextSelectableByMouse)
        self.title_label.setAlignment(Qt.AlignCenter)

        self.input_label = QLabel('Enter a sentence:')
        self.input_label.setObjectName("label_1")
        self.input_field = QTextEdit()
        self.input_field.setTextInteractionFlags(
            self.input_field.textInteractionFlags() | QtCore.Qt.TextSelectableByMouse)

        self.identify_button = QPushButton('Identify Errors')
        self.identify_button.setObjectName("btn_process_data")
        self.identify_button.clicked.connect(self.identify_errors)

        self.error_label = QLabel('Error Identification:')
        self.error_label.setObjectName("label_1")
        self.error_display = QLabel('')
        self.error_display.setTextInteractionFlags(
            self.error_display.textInteractionFlags() | QtCore.Qt.TextSelectableByMouse)

        self.correction_label = QLabel('Corrected Sentence:')
        self.correction_label.setObjectName("label_1")
        self.correction_field = QTextEdit()

        self.correct_button = QPushButton('Correct and Recheck')
        self.correct_button.setObjectName("btn_process_data")
        self.correct_button.clicked.connect(self.correct_and_recheck)

        self.search_button = QPushButton('Search for Similar Phrases')
        self.search_button.setObjectName("btn_process_data")
        self.search_button.clicked.connect(self.search_similar_phrases)

        self.add_data_button = QPushButton('Add Data (Admin)')
        self.add_data_button.setObjectName("btn_process_data")
        self.add_data_button.clicked.connect(self.add_data)

        self.predict_label = QLabel('Model Prediction:')
        self.predict_label.setObjectName("label_1")
        self.predict_display = QLabel('')
        self.predict_display.setTextInteractionFlags(
            self.predict_display.textInteractionFlags() | QtCore.Qt.TextSelectableByMouse)

        self.model_button = QPushButton('model Prediction')
        self.model_button.setObjectName("btn_process_data")
        self.model_button.clicked.connect(self.make_prediction)
        self.model_display = QLabel('')
        self.model_display.setTextInteractionFlags(
            self.model_display.textInteractionFlags() | QtCore.Qt.TextSelectableByMouse)

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_field)
        layout.addWidget(self.identify_button)
        layout.addWidget(self.error_label)
        layout.addWidget(self.error_display)
        layout.addWidget(self.correction_label)
        layout.addWidget(self.correction_field)
        layout.addWidget(self.correct_button)
        layout.addWidget(self.search_button)
        layout.addWidget(self.predict_display)
        layout.addWidget(self.model_button)
        layout.addWidget(self.predict_label)
        layout.addWidget(self.model_display)
        layout.addWidget(self.add_data_button)
        self.setLayout(layout)
    def identify_errors(self):
        if self.input_field.toPlainText():
            phrase=self.input_field.toPlainText()
            tool = LanguageTool('fr')
            corrections = tool.check(phrase)
            try:
                if corrections:
                    A = ""
                    for i in range(0, len(corrections)):
                        A = A + str(corrections[i]) + "\n"
                    output1 = str(A)
                else:
                    output1 = "Aucune erreur trouvée dans la phrase."
            except:
                if output1 is not str:
                    output1=str(output1)
        else :
            output1 = "ENTER THE PHRASE !"
        self.error_display.setText(output1)
    def correct_and_recheck(self):
        if self.correction_field.toPlainText():
            phrase = self.correction_field.toPlainText()
            tool = LanguageTool('fr')
            corrections = tool.check(phrase)
            try:
                if corrections:
                    A = ""
                    for i in range(0, len(corrections)):
                        A = A + str(corrections[i]) + "\n"
                    output1 = str(A)
                else:
                    output1 = "Aucune erreur trouvée dans la phrase."
            except:
                if output1 is not str:
                    output1 = str(output1)
        else:
            output1 = "ENTER THE PHRASE THE CORRECT PHRASE !"
        self.error_display.setText(output1)

    def search_similar_phrases(self):
        input_file="POLUXDATAfr_FR.TUU"
        if self.correction_field.toPlainText() :
            input_sentence=self.correction_field.toPlainText()
            with open(input_file, "r", encoding="utf-8") as file:
                sentences_list = file.readlines()
                phrases_list = []
                for line in sentences_list:
                    sentence = line.split(";")[1].strip()
                    phrases_list.append(sentence)
            phrases_list_net = [sentence.strip() for sentence in phrases_list]
            similar_scores = []
            for sentence in phrases_list_net:
                similarity_score = calculer_similarite(input_sentence, sentence)
                similar_scores.append(similarity_score)

            most_similar_index = np.argmax(similar_scores)
            most_similar_sentence = phrases_list_net[most_similar_index]
            V = similar_scores[most_similar_index]*100
            V="%.2f"%V +' %'
            num_before_semicolon = sentences_list[most_similar_index].split(";")[0].strip()
            output2 ='the similare: \n'+str([num_before_semicolon, most_similar_sentence, V])
            self.predict_display.setText(output2)
        else :
            output2 = "Pas de phrase dans le champ de phrase correcte !"
            self.predict_display.setText(output2)
    def add_data(self):
        password, ok = QInputDialog.getText(self, 'Admin Password', 'Enter admin password:')
        if ok and password == 'yassin':  # Use the correct password here
            add_data_dialog = AddDataDialog()
            add_data_dialog.exec_()
        else:
            QMessageBox.warning(self, 'Access Denied', 'Incorrect password. Access denied.')
    def make_prediction(self, sentence):
        if self.correction_field.toPlainText() :
            ph = self.correction_field.toPlainText()
            predicted_percentage_1 = prediction(ph)
            predicted_percentage_1 = predicted_percentage_1 * 100
            predicted_percentage_1 = "%.2f" % predicted_percentage_1
            predicted_percentage_1 = "The sentence will be valid with the percentage of: " + str(predicted_percentage_1) + "%"
            output = str(predicted_percentage_1)
        else :
            output = "Pas de phrase dans le champ de phrase correcte !"
        self.model_display.setText(output)

class AddDataDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()
    def init_ui(self):
        self.setWindowTitle('Add Data')
        self.setGeometry(300, 300, 400, 300)

        self.upload_button = QPushButton('retrain_model')
        self.upload_button.clicked.connect(self.retrain_model)

        self.btn_choose_file_b = QPushButton("Choose the text file :", self)
        self.btn_choose_file_b.clicked.connect(self.choose_file_b)


        self.data_field = QTextEdit()
        self.valid_radio = QRadioButton('Valid')
        self.not_valid_radio = QRadioButton('Not Valid')
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.valid_radio)
        self.button_group.addButton(self.not_valid_radio)

        self.add_button = QPushButton('Add Data')
        self.add_button.clicked.connect(self.add_data)

        self.etat_data = QLabel('')
        self.etat_data.setTextInteractionFlags(
            self.etat_data.textInteractionFlags() | QtCore.Qt.TextSelectableByMouse)


        self.layout = QFormLayout()
        self.layout.addWidget(self.btn_choose_file_b)
        self.layout.addWidget(self.valid_radio)
        self.layout.addWidget(self.not_valid_radio)
        self.layout.addWidget(self.add_button)
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.etat_data)

        self.setLayout(self.layout)
    def choose_file_b(self):
        global file_name
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose the text file : ", '', 'text File (*.txt)', options=options)
        if file_name:
            self.file_b = file_name
            output3 = "Le fichier texte est charger avec succés"
            self.etat_data.setText(output3)
        else :
            output3 = "Attention choisir le fichier texte !"
            self.etat_data.setText(output3)
    def retrain_model(self):
            df = pd.read_excel('DATA_TOTAL_Diminue_%P_pret_1.xlsx')
            sentences = df['Phrase'].tolist()
            labels = df['Label'].tolist()
            # Instancier le Tokenizer et ajuster sur les phrases
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(sentences)
            # Convertir les phrases en séquences numériques
            sequences = tokenizer.texts_to_sequences(sentences)
            # Obtenir le nombre total de mots dans le vocabulaire
            vocab_size = len(tokenizer.word_index) + 1
            # Paddings pour que toutes les séquences aient la même longueur
            max_sequence_length = max(len(seq) for seq in sequences)
            padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
            labels = np.array(labels)
            np.random.seed(42)
            tf.random.set_seed(42)
            # Créer le modèle
            model = Sequential()
            model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
            model.add(LSTM(64))
            model.add(Dense(1, activation='linear'))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.summary()

            model.fit(padded_sequences, labels, epochs=10, batch_size=32)
            model.save("Validation_Check_Model.h5")
    def add_data(self):
        if (self.valid_radio.isChecked() or self.not_valid_radio.isChecked()) and self.file_b :
            label = "Valid" if self.valid_radio.isChecked() else "Not Valid"
            if label == "Valid":
                excel_file_path = "DATA_TOTAL_Diminue_%P_pret_1.xlsx"
                txt_file_path = file_name
                workbook = openpyxl.load_workbook(excel_file_path)
                sheet = workbook.active
                with open(txt_file_path, "r", encoding='utf-8') as txt_file:
                    lines = txt_file.readlines()
                for line in lines:
                    cell = sheet["A{}".format(sheet.max_row + 1)]
                    cell.value = line.strip()
                    cell_1 = sheet["B{}".format(sheet.max_row)]
                    cell_1.value = random.uniform(0.9, 1)
                workbook.save(excel_file_path)
            else :
                excel_file_path = "DATA_TOTAL_Diminue_%P_pret_1.xlsx"
                txt_file_path = file_name
                workbook = openpyxl.load_workbook(excel_file_path)
                sheet = workbook.active
                with open(txt_file_path, "r", encoding='utf-8') as txt_file:
                    lines = txt_file.readlines()
                for line in lines:
                    cell = sheet["A{}".format(sheet.max_row + 1)]
                    cell.value = line.strip()
                    cell_1 = sheet["B{}".format(sheet.max_row)]
                    cell_1.value = random.uniform(0, 0.1)
                workbook.save(excel_file_path)
        else :
            output4 = "Attention : Charger le fichier texte & Choisir le type de phrases !"
            self.etat_data.setText(output4)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SentenceValidationApp()
    window.show()
    sys.exit(app.exec_())
