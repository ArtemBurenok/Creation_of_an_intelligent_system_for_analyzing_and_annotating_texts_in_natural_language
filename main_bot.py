from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from webdriver_manager.chrome import ChromeDriverManager
import time
import telebot
import os
import selenium
from bs4 import BeautifulSoup
import pandas as pd
import spacy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from secret_key import key


tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

MODEL_NAME = 'cointegrated/rut5-base-absum'
model_T5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer_T5 = T5Tokenizer.from_pretrained(MODEL_NAME)

nlp = spacy.load("ru_core_news_lg")
bot = telebot.TeleBot(key)


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.from_user.id, "Скинь ссылку на отзывы к товару, которые ты хочешь проанализировать или \n"
                                           "Excel файл с комментариями (необходим один столбец с названием Comments).")


@bot.message_handler(content_types=['document'])
def file_reply(message):
    file_id = message.document.file_id
    excel_file = bot.get_file(file_id)

    try:
        with open('data.xlsx', 'wb') as save_file:
            save_file.write(bot.download_file(excel_file.file_path))

        data = pd.read_excel('data.xlsx', engine='openpyxl', keep_default_na=True)
        os.remove('data.xlsx')

        if 'Comments' not in data.columns:
            raise KeyError

        bot.send_message(message.chat.id, 'Файл загружен. Анализируем данные.')

        analyser = SentimentIntensityAnalyzer()
        data_list = np.array(data['Comments'][~data['Comments'].isna()])

        positive, negative, neutral = 0, 0, 0
        pos_list, neg_list, neu_list = [], [], []

        for item in data_list:
            scores = analyser.polarity_scores(item)

            if scores['compound'] > 0:
                pos_list.append(item)
                positive += 1
            elif scores['neu'] > 0.8:
                neu_list.append(item)
                neutral += 1
            else:
                neg_list.append(item)
                negative += 1

        annotation_pos, annotation_neg, annotation_neu = (annotation(list(pos_list)), annotation(list(neg_list)),
                                                          annotation(list(neu_list)))

        whole_list = list(pos_list) + list(neg_list) + list(neu_list)

        tematics = clusterization(whole_list)
        string_tematics = " - " + "\n - ".join(tematics)

        bot.send_message(message.chat.id, f'Количество записей: {data.shape[0]} \n'
                                          f'Позитивные: {positive} ({round(positive / data.shape[0] * 100, 2)} %) \n'
                                          f'Нейтральные: {neutral} ({round(neutral / data.shape[0] * 100, 2)} %) \n'
                                          f'Негативные: {negative} ({round(negative / data.shape[0] * 100, 2)} %) \n')

        bot.send_message(message.chat.id, "<b>Аннотация к позитивному тексту:</b> \n"
                                          f"{annotation_pos} \n"
                                          "\n"
                                          f"<b>Аннотация к негативному тексту:</b> \n"
                                          f"{annotation_neg} \n"
                                          "\n"
                                          f"<b>Аннотация к нейтральному тексту:</b> \n"
                                          f"{annotation_neu}", parse_mode="html")

        bot.send_message(message.chat.id, f'Тематическая кластеризация: \n'
                                          f"\n"
                                          f'{string_tematics}')
    except OSError:
        bot.send_message(message.chat.id, 'Скинь, пожалуйста, excel файл.')
    except KeyError:
        bot.send_message(message.chat.id, 'Отсутствует столбец с названием Comments или столбцов несколько.')


@bot.message_handler(content_types=['text'])
def message_reply(message):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.maximize_window()

    try:
        if message.text[-5:] == '.aspx':
            raise ValueError
        driver.get(url=message.text)
        bot.send_message(message.chat.id, 'Собираем данные. Пожалуйста, подожди.')
        time.sleep(0.5)

        sum_location_list = []

        while True:
            find_more_element = driver.find_element(By.TAG_NAME, 'footer')
            location = find_more_element.location

            if len(sum_location_list) >= 15:
                if sum_location_list[-1] == sum_location_list[-15]:
                    sum_location_list = []
                    with open('page.html', 'w') as file:
                        file.write(driver.page_source)
                    bot.send_message(message.chat.id, "Анализируем данные.")
                    break
                else:
                    action = ActionChains(driver)
                    action.move_to_element(find_more_element).perform()

                    sum_location_list.append(location['y'])
                    time.sleep(0.1)
            else:
                action = ActionChains(driver)
                action.move_to_element(find_more_element).perform()

                sum_location_list.append(location['y'])
                time.sleep(0.1)
    except selenium.common.exceptions.InvalidArgumentException:
        bot.send_message(message.chat.id, "Пожалуйста, скинь ссылку")
    except selenium.common.exceptions.WebDriverException:
        bot.send_message(message.chat.id, 'Проверь, пожалуйста, ссылку на ошибки.')

    finally:
        driver.close()
        driver.quit()

    try:
        len_feedback, pos, neu, neg, pos_list, neg_list, neu_list, product_name = get_items_urls('page.html')

        if len_feedback == 0:
            raise ZeroDivisionError

        annotation_pos, annotation_neg, annotation_neu = annotation(pos_list), annotation(neg_list), annotation(neu_list)

        whole_list = pos_list + neg_list + neu_list

        tematics = clusterization(whole_list)

        string_tematics = " - " + "\n - ".join(tematics)

        bot.send_message(message.chat.id, f'Количество записей: {len_feedback} \n'
                                          f'Позитивные: {pos} ({round(pos / len_feedback * 100, 2)} %) \n'
                                          f'Нейтральные: {neu} ({round(neu / len_feedback * 100, 2)} %) \n'
                                          f'Негативные: {neg} ({round(neg / len_feedback * 100, 2)} %) \n')


        bot.send_message(message.chat.id, "<b>Аннотация к позитивному тексту:</b> \n"
                                          f"{annotation_pos} \n"
                                          "\n"
                                          f"<b>Аннотация к негативному тексту:</b> \n"
                                          f"{annotation_neg} \n"
                                          "\n"
                                          f"<b>Аннотация к нейтральному тексту:</b> \n"
                                          f"{annotation_neu}", parse_mode="html")

        bot.send_message(message.chat.id, f'Тематическая кластеризация: \n'
                                          f"\n"
                                          f'{string_tematics}')

        os.remove('page.html')
    except FileNotFoundError:
        pass
    except ZeroDivisionError:
        bot.send_message(message.chat.id, f'На странице было найдено 0 комментариев.')


def get_items_urls(html_file_path):
    with open(html_file_path) as file:
        html_file = file.read()

    feedback_list = []
    soup = BeautifulSoup(html_file, 'lxml')
    items_p = soup.findAll('p', class_='feedback__text')

    name = soup.findAll('a', class_='product-line__name')
    product_name = name[0].text.split('/ ')[1]

    analyser = SentimentIntensityAnalyzer()

    positive, negative, neutral = 0, 0, 0
    pos_list, neg_list, neu_list = [], [], []

    for item in items_p:
        feedback_list.append(item.text)
        scores = analyser.polarity_scores(item.text)

        if scores['compound'] > 0:
            pos_list.append(item.text)
            positive += 1
        elif scores['neu'] > 0.8:
            neu_list.append(item.text)
            neutral += 1
        else:
            neg_list.append(item.text)
            negative += 1

    return len(feedback_list), positive, neutral, negative, pos_list, neg_list, neu_list, product_name


def annotation(list_comments):
    text_pos = " ".join(list_comments)

    def summarize(text, n_words=None, compression=None, max_length=1000, num_beams=3, do_sample=False,
                  repetition_penalty=10.0, **kwargs):
        if n_words:
            text = '[{}] '.format(n_words) + text
        elif compression:
            text = '[{0:.1g}] '.format(compression) + text
        # x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
        x = tokenizer_T5(text, return_tensors='pt', padding=True)
        with torch.inference_mode():
            out = model_T5.generate(
                **x,
                max_length=max_length, num_beams=num_beams,
                do_sample=do_sample, repetition_penalty=repetition_penalty,
                **kwargs
            )
        return tokenizer_T5.decode(out[0], skip_special_tokens=True)

    return summarize(text_pos)


def clusterization(list_comments):
    sentences = list_comments
    embeddings_list = []

    for s in sentences:
        encoded_input = tokenizer(s, padding=True, truncation=True, max_length=64, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embedding = model_output.pooler_output
        embeddings_list.append((embedding)[0].numpy())

    embeddings = np.asarray(embeddings_list)

    def determine_k(embeddings):
        k_min = 2

        clusters = [x for x in range(2, k_min * 11)]

        metrics = []

        for i in clusters:
            metrics.append((KMeans(n_clusters=i).fit(embeddings)).inertia_)

        k = elbow(k_min, clusters, metrics)
        return k

    def elbow(k_min, clusters, metrics):
        score = []

        for i in range(k_min, clusters[-3]):
            y1 = np.array(metrics)[:i + 1]
            y2 = np.array(metrics)[i:]

            df1 = pd.DataFrame({'x': clusters[:i + 1], 'y': y1})
            df2 = pd.DataFrame({'x': clusters[i:], 'y': y2})

            reg1 = LinearRegression().fit(np.asarray(df1.x).reshape(-1, 1), df1.y)
            reg2 = LinearRegression().fit(np.asarray(df2.x).reshape(-1, 1), df2.y)

            y1_pred = reg1.predict(np.asarray(df1.x).reshape(-1, 1))
            y2_pred = reg2.predict(np.asarray(df2.x).reshape(-1, 1))

            score.append(mean_squared_error(y1, y1_pred) + mean_squared_error(y2, y2_pred))

        return np.argmin(score) + k_min

    k_opt = determine_k(embeddings)
    kmeans = KMeans(n_clusters=k_opt, random_state=42).fit(embeddings)
    kmeans_labels = kmeans.labels_

    data_new = pd.DataFrame()
    data_new['text'] = sentences
    data_new['label'] = kmeans_labels
    data_new['embedding'] = list(embeddings)

    kmeans_centers = kmeans.cluster_centers_
    top_texts_list = []

    for i in range(0, k_opt):
        cluster = data_new[data_new['label'] == i]
        embeddings = list(cluster['embedding'])
        texts = list(cluster['text'])
        distances = [euclidean_distances(kmeans_centers[0].reshape(1, -1), e.reshape(1, -1))[0][0] for e in embeddings]
        scores = list(zip(texts, distances))
        top_3 = sorted(scores, key=lambda x: x[1])[:3]
        top_texts = list(zip(*top_3))[0]
        top_texts_list.append(top_texts)

    def summarize(text, n_words=None, compression=None, max_length=1000, num_beams=3, do_sample=False, repetition_penalty=10.0, **kwargs):
        if n_words:
            text = '[{}] '.format(n_words) + text
        elif compression:
            text = '[{0:.1g}] '.format(compression) + text
        # x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
        x = tokenizer_T5(text, return_tensors='pt', padding=True)
        with torch.inference_mode():
            out = model_T5.generate(
                **x,
                max_length=max_length, num_beams=num_beams,
                do_sample=do_sample, repetition_penalty=repetition_penalty,
                **kwargs
            )
        return tokenizer_T5.decode(out[0], skip_special_tokens=True)

    summ_list = []
    for top in top_texts_list:
        summ_list.append(summarize(' '.join(list(top))))

    return summ_list


bot.infinity_polling(none_stop=True, interval=0, timeout=10, long_polling_timeout=5)
