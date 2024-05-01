import os
import time
import wave
import shutil
import logging

from telegram import ReplyKeyboardRemove, Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from utils import db, DATA_DIR


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


TRAIN_LANGUAGE, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_MODEL_NAME, TRAIN_DATASET = range(5)


async def train_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    os.makedirs(os.path.join(DATA_DIR, str(update.message.from_user.id)), exist_ok=True)
    await update.message.reply_text(
        "Выберите язык модели",
        reply_markup=ReplyKeyboardMarkup([["Русский", "Английский"]], one_time_keyboard=True)
    )
    with db.connect() as con:
        cursor = con.cursor()
        cursor.execute("delete from train_params where user_id = ?", (update.message.from_user.id, ))
        con.commit()
        cursor.execute("insert into train_params (user_id) values (?)", (update.message.from_user.id, ))
        con.commit()
    return TRAIN_LANGUAGE


async def train_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    language = update.message.text
    if language == "Русский":
        pretrain_name = "SnowieV3"
    elif language == "Английский":
        pretrain_name = "Titan"
    else:
        await update.message.reply_text("Выбран неверный язык. Повторите попытку")
        return TRAIN_LANGUAGE

    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "update train_params set pretrain_name = ? where user_id = ?",
            (pretrain_name, update.message.from_user.id)
        )
        con.commit()

    await update.message.reply_text(
        "Выберите количество эпох для обучения модели",
        reply_markup=ReplyKeyboardRemove()
    )
    return TRAIN_EPOCHS


async def train_epochs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    epochs = update.message.text
    try:
        epochs = int(epochs.strip())
    except:
        await update.message.reply_text("Количество эпох - целое число. Повторите попытку.")
        return TRAIN_EPOCHS

    if epochs < 1:
        await update.message.reply_text("Минимальные число эпох - 1. Повторите попытку")
        return TRAIN_EPOCHS
    elif epochs > 250:
        await update.message.reply_text("Максимальное число эпох - 250. Повторите попытку")
        return TRAIN_EPOCHS

    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "update train_params set epochs = ? where user_id = ?",
            (epochs, update.message.from_user.id)
        )
        con.commit()

    await update.message.reply_text(
        "Выберите размер батча для обучения модели"
    )

    return TRAIN_BATCH_SIZE


async def train_batch_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    batch_size = update.message.text
    try:
        batch_size = int(batch_size.strip())
    except:
        await update.message.reply_text("Размер батча - целое число. Повторите попытку.")
        return TRAIN_BATCH_SIZE

    if batch_size < 1:
        await update.message.reply_text("Минимальный размер батча - 1. Повторите попытку")
        return TRAIN_BATCH_SIZE
    elif batch_size > 12:
        await update.message.reply_text("Максимальный размер батча - 12. Повторите попытку")
        return TRAIN_BATCH_SIZE

    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "update train_params set batch_size = ? where user_id = ?",
            (batch_size, update.message.from_user.id)
        )
        con.commit()

    await update.message.reply_text(
        "Укажите название модели"
    )

    return TRAIN_MODEL_NAME


async def train_model_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    model_name = update.message.text
    if not model_name.isalnum():
        await update.message.reply_text("Название модели может содержать только цифры и буквы. Повторите попытку")
        return TRAIN_MODEL_NAME

    if len(model_name) > 32:
        await update.message.reply_text("Длина названия модели не может превышать 32 символа. Повторите попытку")
        return TRAIN_MODEL_NAME

    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "select count(*) from models where user_id = ? and model_name = ?",
            (model_name, update.message.from_user.id)
        )
        if curs.fetchone()[0]:
            await update.message.reply_text("У вас уже есть модель с таким названием. Повторите попытку")
            return TRAIN_MODEL_NAME

        curs.execute(
            "update train_params set model_name = ? where user_id = ?",
            (model_name, update.message.from_user.id)
        )
        con.commit()
        curs.execute(
            "delete from train_data where user_id = ? and model_name = ?",
            (update.message.from_user.id, model_name)
        )
        con.commit()
        shutil.rmtree(os.path.join(DATA_DIR, str(update.message.from_user.id), model_name), ignore_errors=True)

    await update.message.reply_text(
        "Отправьте wav файл с аудио для обучения. Ограничение на файл - 20 мегабайт. "
        "При необходимости использования большого датасета разделите исходный wav файл на несколько"
    )
    return TRAIN_DATASET


async def train_dataset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    attachment = update.message.effective_attachment
    reply_keyboard = [
        ["Начать обучение"]
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    with db.connect() as con:
        curs = con.cursor()
        curs.execute("select model_name from train_params where user_id = ?", (update.message.from_user.id, ))
        [model_name] = curs.fetchone()
    if attachment is None:
        if update.message.text != "Начать обучение":
            return TRAIN_DATASET

        with db.connect() as con:
            curs = con.cursor()
            curs.execute("select count(*) from train_data where user_id = ? and model_name = ?",
                         (update.message.from_user.id, model_name))
            if not curs.fetchone()[0]:
                await update.message.reply_text(
                    "Перед началом задачи отправьте wav файл, который будет использоваться для обучения модели",
                    reply_markup=ReplyKeyboardRemove()
                )
                return TRAIN_DATASET
            now = int(time.time())
            curs.execute(
                "insert into queue (user_id, model_name, status, add_time, task_type) values (?, ?, ?, ?, ?)",
                (update.message.from_user.id, model_name, "queue", now, "train")
            )
            con.commit()
        await update.message.reply_text("Задача обучения добавлена в очередь", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    if not attachment.file_name.endswith(".wav"):
        await update.message.reply_text("Название файла должно оканчиваться на .wav", reply_markup=markup)
        return TRAIN_DATASET

    os.makedirs(os.path.join(DATA_DIR, str(update.message.from_user.id), model_name), exist_ok=True)
    try:
        file = await attachment.get_file()
    except:
        await update.message.reply_text(
            "wav файл слишком большой. Разделите его на треки весящие меньше 20 мегабайт и повторите попытку",
            reply_markup=markup
        )
        return TRAIN_DATASET

    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "select count(*) from train_data where user_id = ? and model_name = ?",
            (update.message.from_user.id, model_name)
        )
        [file_index] = curs.fetchone()
    if file_index >= 5:
        await update.message.reply_text(
            "В датасете может быть не более 5 файлов.",
            reply_markup=markup
        )
        return TRAIN_DATASET

    file_path = os.path.join(DATA_DIR, str(update.message.from_user.id), model_name, str(file_index)) + ".wav"
    await file.download_to_drive(file_path)

    try:
        with wave.open(file_path):
            pass
    except wave.Error:
        await update.message.reply_text("Убедитесь что вы передали валидный wav файл", reply_markup=markup)
        return TRAIN_DATASET

    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "insert into train_data (user_id, model_name, filename) values (?, ?, ?)",
            (update.message.from_user.id, model_name, f"{file_index}.wav")
        )
        con.commit()

    file_msg = f"Вы можете отправить еще {4 - file_index} файла."
    if 3 == file_index:
        file_msg = "Вы можете отправить еще 1 файл."
    elif 4 == file_index:
        file_msg = "В датасет больше нельзя добавить файлы"
    await update.message.reply_text(
        f"Файл добавлен. {file_msg}",
        reply_markup=markup
    )
    return TRAIN_DATASET


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    await update.message.reply_text(
        "Bye! I hope we can talk again some day.", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("6863679219:AAFXkmmvYQ988Cy2MogkDgVygUUAV4V1kAM").build()
    train_handler = ConversationHandler(
        entry_points=[CommandHandler("train_model", train_model)],
        states={
            TRAIN_LANGUAGE: [MessageHandler(filters.TEXT, train_language)],
            TRAIN_EPOCHS: [MessageHandler(filters.TEXT, train_epochs)],
            TRAIN_BATCH_SIZE: [MessageHandler(filters.TEXT, train_batch_size)],
            TRAIN_MODEL_NAME: [MessageHandler(filters.TEXT, train_model_name)],
            TRAIN_DATASET: [MessageHandler(filters.AUDIO | filters.TEXT, train_dataset)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(train_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()