import os
import time
import uuid
import wave
import shutil
import logging
import datetime
from functools import wraps


import matplotlib.pyplot as plt
from telegram import ReplyKeyboardRemove, Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from utils import db, DATA_DIR, valid_model_name, token


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


TRAIN_PRETRAIN, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_MODEL_NAME, TRAIN_DATASET = range(5)
INFER_MODEL, INFER_PITCH = range(2)


def cancel_handler(f):
    @wraps(f)
    async def inner(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message.text == "/cancel":
            await update.message.reply_text("Задача отменена", reply_markup=ReplyKeyboardRemove())
            return ConversationHandler.END
        return await f(update, context)
    return inner


@cancel_handler
async def train_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    os.makedirs(os.path.join(DATA_DIR, str(update.message.from_user.id)), exist_ok=True)
    with db.connect() as con:
        curs = con.cursor()
        curs.execute("select pretrain_name, sr from pretrains")
        res = curs.fetchall()
    if not res:
        await update.message.reply_text(
            "В сервисе нет pretrain моделей. Зайдите позже!"
        )
        return ConversationHandler.END

    markup = []
    for i in range(0, len(res), 2):
        m = [f"{res[i][0]}; SR={res[i][1]}HZ"]
        if i + 1 < len(res):
            m.append(f"{res[i+1][0]}; SR={res[i][1]}HZ")
        markup.append(m)
    await update.message.reply_text(
        "Выберите pretrain модель",
        reply_markup=ReplyKeyboardMarkup(markup, one_time_keyboard=True)
    )
    with db.connect() as con:
        cursor = con.cursor()
        cursor.execute("delete from train_params where user_id = ?", (update.message.from_user.id, ))
        con.commit()
        cursor.execute("insert into train_params (user_id) values (?)", (update.message.from_user.id, ))
        con.commit()
    return TRAIN_PRETRAIN


@cancel_handler
async def train_pretrain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    pretrain = update.message.text
    if "; SR=" not in pretrain or not pretrain.endswith("HZ"):
        await update.message.reply_text("Выберите pretrain модель из вариантов на клавиатуре")
        return TRAIN_PRETRAIN
    pretrain_name, _ = pretrain.split(";")

    with db.connect() as con:
        curs = con.cursor()
        curs.execute("select count(*) from pretrains where pretrain_name = ?", (pretrain_name, ))
        if not curs.fetchone()[0]:
            await update.message.reply_text("Выберите pretrain модель из вариантов на клавиатуре")
            return TRAIN_PRETRAIN
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


@cancel_handler
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


@cancel_handler
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


@cancel_handler
async def train_model_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    model_name = update.message.text
    if not valid_model_name(model_name):
        await update.message.reply_text("Название модели может содержать только цифры и латинские буквы. Повторите попытку")
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


@cancel_handler
async def train_dataset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    attachment = update.message.effective_attachment
    reply_keyboard = [
        ["Начать обучение"]
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "select model_name, epochs, pretrain_name, batch_size from train_params where user_id = ?",
            (update.message.from_user.id, )
        )
        [model_name, epochs, pretrain_name, batch_size] = curs.fetchone()
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

            curs.execute(
                "insert into models (user_id, model_name, public, pretrain_name, epochs, batch_size) values"
                "(?, ?, ?, ?, ?, ?)",
                (update.message.from_user.id, model_name, False, pretrain_name, epochs, batch_size)
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
    await update.message.reply_text(
        "Задача отменена", reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


async def dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "select model_name, status, add_time, task_type from queue where user_id = ?",
            (update.message.from_user.id, )
        )
        res = curs.fetchall()
    if not res:
        await update.message.reply_text("У вас нет задач")
        return
    mes = "\n".join(
        f"[{i + 1}] <{model_name}> {{{status}}} ({task_type}) {datetime.datetime.fromtimestamp(add_time)}"
        for i, (model_name, status, add_time, task_type) in enumerate(res)
    )
    await update.message.reply_text(mes)


@cancel_handler
async def infer_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    with db.connect() as con:
        curs = con.cursor()
        curs.execute("select model_name from models where user_id = ?", (update.message.from_user.id, ))
        res = curs.fetchall()
        if not res:
            await update.message.reply_text("У вас нет моделей для преобразования голоса")
            return ConversationHandler.END

        curs.execute(
            "select count(*) from queue where user_id = ? and task_type = ? and status not in ('error', 'done')",
            (update.message.from_user.id, "infer")
        )
        if curs.fetchone()[0] >= 3:
            await update.message.reply_text("У вас уже запущено три задачи преобразования голоса. Подождите")
            return ConversationHandler.END

    attachment = update.message.effective_attachment
    file = await attachment.get_file()
    filename = uuid.uuid4().hex + ".ogg"
    os.makedirs(os.path.join(DATA_DIR, "infer", str(update.message.from_user.id)), exist_ok=True)
    await file.download_to_drive(os.path.join(DATA_DIR, "infer", str(update.message.from_user.id), filename))
    with db.connect() as con:
        curs = con.cursor()
        curs.execute("delete from infers where user_id = ?", (update.message.from_user.id,))
        con.commit()
        curs.execute("insert into infers (user_id, infer_path) values (?, ?)", (update.message.from_user.id, filename))
        con.commit()
    markup = []
    for i in range(0, len(res), 2):
        m = [res[i][0]]
        if i + 1 < len(res):
            m.append(res[i + 1][0])
        markup.append(m)
    await update.message.reply_text("Выберите модель", reply_markup=ReplyKeyboardMarkup(markup))
    return INFER_MODEL


@cancel_handler
async def infer_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    model_name = update.message.text
    with db.connect() as con:
        curs = con.cursor()
        curs.execute(
            "select count(*) from models where user_id = ? and model_name = ?",
            (update.message.from_user.id, model_name)
        )
        if not curs.fetchone()[0]:
            await update.message.reply_text("Выбрана неверная модель")
            return INFER_MODEL

        curs.execute("update infers set model_name = ? where user_id = ?", (model_name, update.message.from_user.id))
        con.commit()
    await update.message.reply_text(
        "Выберите изменение тона. Если у вас высокий голос, а хотите сделать низкий, "
        "то выбирайте отрицательное значение. Если хотите сделать наоборот, выбирайте положительное. "
        "Чем больше значение, тем сильнее изменение. Допустимые значения от -24 до 24. "
        "Если не хотите менять тон выберите 0",
        reply_markup=ReplyKeyboardMarkup([["0"]])
    )
    return INFER_PITCH


@cancel_handler
async def infer_pitch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    pitch = update.message.text
    try:
        pitch = int(pitch)
    except:
        await update.message.reply_text("Значение изменения тона должно быть целом числом от -24 до 24")
        return INFER_PITCH

    if not -24 <= pitch <= 24:
        await update.message.reply_text("Значение изменения тона должно быть целом числом от -24 до 24")
        return INFER_PITCH

    with db.connect() as con:
        curs = con.cursor()
        curs.execute("select infer_path, model_name from infers where user_id = ?", (update.message.from_user.id,))
        [infer_path, model_name] = curs.fetchone()
        now = int(time.time())
        curs.execute(
            "insert into queue (user_id, model_name, status, add_time, task_type, infer_path, f0up) values "
            "(?, ?, ?, ?, ?, ?, ?)", (update.message.from_user.id, model_name, "queue", now, "infer", infer_path, pitch)
        )
        con.commit()

    await update.message.reply_text("Задача добавлена в очередь", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


def main() -> None:
    application = Application.builder().token(token).build()
    train_handler = ConversationHandler(
        entry_points=[CommandHandler("train_model", train_model)],
        states={
            TRAIN_PRETRAIN: [MessageHandler(filters.TEXT, train_pretrain)],
            TRAIN_EPOCHS: [MessageHandler(filters.TEXT, train_epochs)],
            TRAIN_BATCH_SIZE: [MessageHandler(filters.TEXT, train_batch_size)],
            TRAIN_MODEL_NAME: [MessageHandler(filters.TEXT, train_model_name)],
            TRAIN_DATASET: [MessageHandler(filters.AUDIO | filters.TEXT, train_dataset)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    infer_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.VOICE, infer_start)],
        states={
            INFER_MODEL: [MessageHandler(filters.TEXT, infer_model)],
            INFER_PITCH: [MessageHandler(filters.TEXT, infer_pitch)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    application.add_handler(train_handler)
    application.add_handler(infer_handler)
    application.add_handler(CommandHandler("dashboard", dashboard))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
