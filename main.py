from telegram.ext import ApplicationBuilder, MessageHandler, filters, CallbackQueryHandler, CommandHandler, CallbackContext, Updater
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from gpt import *
import queue
#from telegram.ext.updater import Queue
from util import *
import sqlite3
from aiogram import Bot, Dispatcher, types
import tensorflow as tf
import os
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import easyocr
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import openai
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import httpx as httpx
# Создаем или подключаем базу данных
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY)')
# model = Sequential()
# model = VGG16(weights='C:/Users/user/Desktop/Проект PhotoConAI/Шаблон/model.weights.h5', include_top=False)
# model.load_weights('C:/Users/user/Desktop/Проект PhotoConAI/Шаблон/model.weights.h5')  # Замените на путь к вашей модели
import tkinter as tk
from tkinter import filedialog

async def start(update, context):
    user_id = update.effective_user.id
    c.execute('INSERT OR IGNORE INTO users (user_id) VALUES (?)', (user_id,))
    conn.commit()

    dialog.mode = 'main'
    text = load_message('main')
    await send_text(update, context, text)
    await show_main_menu(update, context, {
        'start': "Запустить",
        "question": "Спросить",
        'don': "Поддержать",
        'count_users': "Количество пользователей",
    })
    keyboard = [
        # [InlineKeyboardButton("Запустить", callback_data='start')],
        [InlineKeyboardButton("Спросить", callback_data='question')],
        [InlineKeyboardButton("Поддержать", callback_data='don')],
        [InlineKeyboardButton("Количество пользователей", callback_data='count_users')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите действие:", reply_markup=reply_markup)
BOT_TOKEN = "#"
update_queue = queue.Queue()
updater = Updater(BOT_TOKEN, update_queue)
# dispatcher = updater.dispatcher

async def question(update, context):
    # dialog.mode = 'gpt'
    await send_text(update, context, "Отправляйте фото с подписью")
    photo = update.message.photo
    if photo:
        id_photo = photo[-1].file_id
        name_photo = update.message.caption
        await send_text(update, context, "1")

        update.message.reply_text(f"Получено фото с id: {id_photo}")
        file = context.bot.get_file(photo[-1].file_id)
        file.download('photo.jpg')

        # Обработка изображения с помощью VGG16
        model = VGG16(weights='imagenet', include_top=False)
        model.load_weights('C:/Users/user/Desktop/Проект PhotoConAI/Шаблон/model.weights.h5')
        img = image.load_img('photo.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)

        # Обработка признаков
        # ... (например, кластеризация или другие операции)

        # Используем GPT для получения ответа
        openai.api_key = '#'
        prompt = f"Опиши, что изображено на фотографии. Используй контекст: '{features}'."
        gpt_response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7,
        )
        gpt_text = gpt_response.choices[0].text.strip()

        # Отправляем ответ
        update.message.reply_text(f"Обработано фото с помощью нейросети!\n\n{gpt_text}")

   #  photo = update.message.photo
   #  id_photo = photo.file_id
   #  name_photo = update.message.caption
   #
   #  # Делаем что-нибудь с id_photo и name_photo
   #  update.message.reply_text(f"Получено фото с id: {id_photo}")
   #  file = context.bot.get_file(photo.file_id)
   #  file.download('photo.jpg')
   #
   #  # Модель
   # # model = VGG16(weights='C:/Users/user/Desktop/Проект PhotoConAI/Шаблон/model.weights.h5', include_top=False)
   #  model = VGG16(weights='imagenet', include_top=False)
   #  model.load_weights('C:/Users/user/Desktop/Проект PhotoConAI/Шаблон/model.weights.h5')
   #  # model.load_weights('C:/Users/user/Desktop/Проект PhotoConAI/Шаблон/model.weights.h5')  # Замените на путь к вашей модели
   #
   #  # Загружаем и предподготавливаем изображение
   #  img = image.load_img('photo.jpg', target_size=(224, 224))
   #  x = image.img_to_array(img)
   #  x = np.expand_dims(x, axis=0)
   #  x = preprocess_input(x)
   #
   #  # Делаем предсказание с помощью модели
   #  features = model.predict(x)
   #  # Обрабатываем полученные признаки (features)
   #  # Например, можно вывести их на экран или выполнить кластеризацию
   #  print(features)
   #
   #  # Отправляем ответное сообщение
   #  update.message.reply_text(f"Обработано фото с помощью нейросети!")
   #  response = {
   #      features
   #  }
   #  # image_text = photo('modeling.weights.h5')
   #  #await handle_photo(update, context, image)
   #  #text = send_text('gpt')
   #  # await handle_photo(update, context, text)
   #  openai.api_key = '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'
   #  prompt = f"Сделай решение задач"
   #  gpt_response = openai.Completion.create(
   #      engine="text-davinci-003",
   #      prompt=prompt,
   #      max_tokens=100,
   #      n=1,
   #      stop=None,
   #      temperature=0.7,
   #  )
   #  gpt_text = gpt_response.choices[0].text.strip()
   #
   #  # Отправляем ответное сообщение
   #  update.message.reply_text(f"Обработано фото с помощью нейросети! {response}\n\n{gpt_text}")
   #

# Регистрируем обработчик фото
# dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))

# await send_text(update, context, image, text)

async def don(update, context):
    await send_text(update, context, "Если вы хотите поддержать, то вот вам варианты помощи:")

async def gpt_dialog(update, context):
    text = update.message.text
    prompt = load_prompt('gpt')
    await send_text(update, context, "Сейчас разберемся🧠")
    answer = await chatgpt.send_question(prompt, text)
    await send_text(update, context, answer)
async def show_user_count(update, context):
    query = update.callback_query
    count = count_users()
    await query.answer()
    await query.edit_message_text(text=f'Количество уникальных пользователей: {count}')
def count_users():
    c.execute('SELECT COUNT(*) FROM users')
    return c.fetchone()[0]

async def hello(update, context):
    if dialog.mode == 'gpt':
        await gpt_dialog(update, context)
    else:
        text = load_message('main')
        await send_photo(update, context, "main")
        await send_text(update, context, text)

async def show_user_count(update, context):
    count = count_users()
    await send_text(update, context, f'Количество уникальных пользователей: {count}')
def count_users():
    c.execute('SELECT COUNT(*) FROM users')
    count = c.fetchone()[0]
    #print(f'Количество пользователей: {count}')

dialog = Dialog()
dialog.mode = None
dialog.list = []

chatgpt = ChatGptService(token='#')  # Вставьте свой ключ

app = ApplicationBuilder().token("#").build()  # Вставьте токен ТГ
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("question", question))
app.add_handler(CommandHandler("don", don))
app.add_handler(CommandHandler("count_users", show_user_count))
app.add_handler(CallbackQueryHandler(show_user_count, pattern='count_users'))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, hello))

app.run_polling()

#count_users()  # Выводим количество уникальных пользователей после старта бота
