import re
import sys
from nltk import TweetTokenizer
from scipy.stats import kurtosis, skew
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from text_processing import TextProcessing
from utils import Utils
from lexical_features import lexical_es, lexical_en


class FeatureExtraction(BaseEstimator, TransformerMixin):


    def __init__(self, lang='es'):
        self.lang = lang  
        try:
            self.lexical = lexical_es if lang == 'es' else lexical_en
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error FeatureExtraction: {0}'.format(e))

    def fit(self, x, y=None):
        return self

    def transform(self, list_messages):
            try:
                # Si se pasa un solo string, se envuelve en una lista
                if isinstance(list_messages, str):
                    list_messages = [list_messages]
                    
                # Procesar cada mensaje de la lista
                features_list = []
                for msg in list_messages:
                    feat = self.get_features(msg)
                    if feat is not None:
                        features_list.append(feat)
                    else:
                        # En caso de falla, genera un vector lleno de ceros
                        features_list.append(np.zeros(35, dtype=np.float32))
                
                return np.array(features_list, dtype=np.float32)
            except Exception as e:
                Utils.standard_error(sys.exc_info())
                print('Error transform: {0}'.format(e))
            return None
            
    def get_features(self, messages: str):
        try:
            features = list(abs(self.get_features_lexical(messages)))
            result = np.array(features, dtype=np.float32)
            return result
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_features: {0}'.format(e))
            return None

    def get_features_lexical(self, message):
        result = None
        try:
            lexical = self.lexical
            text_tokenizer = TweetTokenizer()
            
            # MODIFICACIÓN APLICADA PARTE 1:
            # Se ajustan las etiquetas para que coincidan con las marcas '[mention]', '[url]', etc.
            # que genera TextProcessing.transformer en minúsculas.
            tags = ('[mention]', '[url]', '[hashtag]', '[emoji]', 'rt', 'mention', 'url', 'hashtag', 'emoji')
            vector = dict()
            
            # Preservamos los tokens originales (con sus mayúsculas originales) para detectar gritado
            raw_tokens = text_tokenizer.tokenize(str(message))
            tokens_text = text_tokenizer.tokenize(str(message).lower())
            
            if len(tokens_text) > 0:
                vector['weighted_position'], vector['weighted_normalized'] = self.weighted_position(tokens_text)

                # MODIFICACIÓN APLICADA PARTE 2:
                # Búsqueda corregida para contar adecuadamente con o sin corchetes.
                vector['label_mention'] = float(sum(1 for word in tokens_text if word in ('[mention]', 'mention')))
                vector['label_url'] = float(sum(1 for word in tokens_text if word in ('[url]', 'url')))
                vector['label_hashtag'] = float(sum(1 for word in tokens_text if word in ('[hashtag]', 'hashtag')))
                vector['label_emoji'] = float(sum(1 for word in tokens_text if word in ('[emoji]', 'emoji')))
                vector['label_retweets'] = float(sum(1 for word in tokens_text if word in ('rt', '[rt]')))

                vector['lexical_diversity'] = self.lexical_diversity(message)

                label_word = vector['label_mention'] + vector['label_url'] + vector['label_hashtag']
                label_word = label_word + vector['label_emoji'] + vector['label_retweets']
                vector['label_word'] = float(len(tokens_text) - label_word)

                vector['first_person_singular'] = float(
                    sum(1 for word in tokens_text if word in lexical['first_person_singular']))
                vector['second_person_singular'] = float(
                    sum(1 for word in tokens_text if word in lexical['second_person_singular']))
                vector['third_person_singular'] = float(
                    sum(1 for word in tokens_text if word in lexical['third_person_singular']))
                vector['first_person_plurar'] = float(
                    sum(1 for word in tokens_text if word in lexical['first_person_plurar']))
                vector['second_person_plurar'] = float(
                    sum(1 for word in tokens_text if word in lexical['second_person_plurar']))
                vector['third_person_plurar'] = float(
                    sum(1 for word in tokens_text if word in lexical['third_person_plurar']))

                vector['avg_word'] = np.nanmean([len(word) for word in tokens_text if word not in tags])
                vector['avg_word'] = vector['avg_word'] if not np.isnan(vector['avg_word']) else 0.0
                vector['avg_word'] = round(vector['avg_word'], 4)

                vector['kur_word'] = kurtosis([len(word) for word in tokens_text if word not in tags])
                vector['kur_word'] = vector['kur_word'] if not np.isnan(vector['kur_word']) else 0.0
                vector['kur_word'] = round(vector['kur_word'], 4)

                vector['skew_word'] = skew(np.array([len(word) for word in tokens_text if word not in tags]))
                vector['skew_word'] = vector['skew_word'] if not np.isnan(vector['skew_word']) else 0.0
                vector['skew_word'] = round(vector['skew_word'], 4)

                # Adverbios
                vector['adverb_neg'] = sum(1 for word in tokens_text if word in lexical['adverb_neg'])
                vector['adverb_neg'] = float(vector['adverb_neg'])

                vector['adverb_time'] = sum(1 for word in tokens_text if word in lexical['adverb_time'])
                vector['adverb_time'] = float(vector['adverb_time'])

                vector['adverb_place'] = sum(1 for word in tokens_text if word in lexical['adverb_place'])
                vector['adverb_place'] = float(vector['adverb_place'])

                vector['adverb_mode'] = sum(1 for word in tokens_text if word in lexical['adverb_mode'])
                vector['adverb_mode'] = float(vector['adverb_mode'])

                vector['adverb_cant'] = sum(1 for word in tokens_text if word in lexical['adverb_cant'])
                vector['adverb_cant'] = float(vector['adverb_cant'])

                vector['adverb_all'] = float(vector['adverb_neg'] + vector['adverb_time'] + vector['adverb_place'])
                vector['adverb_all'] = float(vector['adverb_all'] + vector['adverb_mode'] + vector['adverb_cant'])

                vector['adjetives_neg'] = sum(1 for word in tokens_text if word in lexical['adjetives_neg'])
                vector['adjetives_neg'] = float(vector['adjetives_neg'])

                vector['adjetives_pos'] = sum(1 for word in tokens_text if word in lexical['adjetives_pos'])
                vector['adjetives_pos'] = float(vector['adjetives_pos'])

                vector['who_general'] = sum(1 for word in tokens_text if word in lexical['who_general'])
                vector['who_general'] = float(vector['who_general'])

                vector['who_male'] = sum(1 for word in tokens_text if word in lexical['who_male'])
                vector['who_male'] = float(vector['who_male'])

                vector['who_female'] = sum(1 for word in tokens_text if word in lexical['who_female'])
                vector['who_female'] = float(vector['who_female'])

                # =========================================================================
                # NUEVAS CARACTERÍSTICAS LÉXICAS - FASE 1
                # =========================================================================

                # 1. Elongación de caracteres ("buenooo") y risas ("jajaja", "xd")
                vector['char_elongation'] = float(sum(1 for word in tokens_text if re.search(r'([a-zA-Z])\1{2,}', word)))
                vector['laughter_count'] = float(sum(1 for word in tokens_text if re.search(r'(ja|je|ji|jo|ju){2,}|xd', word)))

                # 2. Proporción de palabras en mayúsculas y repetición de signos de exclamación o interrogación
                vector['uppercase_ratio'] = float(sum(1 for word in raw_tokens if word.isupper() and len(word) > 1) / len(tokens_text))
                vector['exclamation_repeat'] = float(len(re.findall(r'!{2,}|¡{2,}', message)))
                vector['question_repeat'] = float(len(re.findall(r'\?{2,}|¿{2,}', message)))

                # 3. Alcance de la negación (conteo de tokens dentro de una ventana de 3 palabras tras una negación)
                neg_scope = 0
                for idx, word in enumerate(tokens_text):
                    if word in lexical['adverb_neg']:
                        neg_scope += len(tokens_text[idx + 1: idx + 4])
                vector['negation_scope'] = float(neg_scope)

                # 4. Intensificadores y atenuadores (usando claves seguras en el diccionario lexical)
                intensifiers = lexical.get('intensifiers', ['muy', 'más', 'super', 'súper', 're', 'demasiado', 'mucho'])
                attenuators = lexical.get('attenuators', ['algo', 'poco', 'medio', 'apenas', 'casi', 'ligeramente'])
                vector['intensifiers_count'] = float(sum(1 for word in tokens_text if word in intensifiers))
                vector['attenuators_count'] = float(sum(1 for word in tokens_text if word in attenuators))

                # 5. Relación entre palabras positivas y negativas (balance de polaridad)
                pos_count = vector['adjetives_pos']
                neg_count = vector['adjetives_neg']
                vector['polarity_ratio'] = float((pos_count - neg_count) / (pos_count + neg_count + 1e-5))

                # 6. Interjecciones y expresiones coloquiales
                interjections = lexical.get('interjections', ['jaja', 'jajaja', 'xd', 'uf', 'uy', 'ay', 'ole', 'ostia', 'vaya'])
                vector['interjections_count'] = float(sum(1 for word in tokens_text if word in interjections))

                result = np.array(list(vector.values()))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_lexical_features: {0}'.format(e))
        return result

    @staticmethod
    def lexical_diversity(text):
        result = None
        try:
            # MODIFICACIÓN APLICADA PARTE 3:
            # Cambiado el cálculo de diversidad léxica por caracteres a diversidad por palabras (Type-Token Ratio).
            text_tokenizer = TweetTokenizer()
            tokens = text_tokenizer.tokenize(str(text).lower())
            if len(tokens) > 0:
                result = round(len(set(tokens)) / len(tokens), 4)
            else:
                result = 0.0
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error lexical_diversity: {0}'.format(e))
        return result

    @staticmethod
    def weighted_position(tokens_text):
        result = None
        try:
            # MODIFICACIÓN APLICADA PARTE 4:
            # Corregido el cálculo para usar enumerate(tokens_text) en lugar de tokens_text.index(w)
            size = len(tokens_text)
            weighted_words = 0.0
            weighted_normalized = 0.0
            for idx, w in enumerate(tokens_text):
                weighted_words += 1 / (1 + idx)
                weighted_normalized += (1 + idx) / size
            result = (weighted_words, weighted_normalized)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error weighted_position: {0}'.format(e))
        return result