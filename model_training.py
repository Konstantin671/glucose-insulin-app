import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def generate_glucose_insulin_data(samples=10000):
    np.random.seed(42)

    # Генерация входных данных
    glucose = np.random.uniform(4, 10, samples)  # Уровень глюкозы в ммоль/л
    insulin = np.random.uniform(5, 20, samples)  # Уровень инсулина в ед.
    carbs = np.random.uniform(0, 100, samples)  # Количество потребленных углеводов в граммах
    time = np.random.uniform(0, 180, samples)  # Время до прогноза в минутах
    time_of_day = np.random.choice([0, 6, 12, 18], samples)  # Время суток

    # Учёт чувствительности к инсулину в зависимости от времени суток
    insulin_sensitivity_map = {
        0: 0.8,  # Ночь
        6: 1.2,  # Утро
        12: 1.0, # День
        18: 0.9  # Вечер
    }
    insulin_sensitivity = np.array([insulin_sensitivity_map[tod] for tod in time_of_day])

    # Будущая глюкоза: логика постепенного влияния углеводов, инсулина и времени
    carb_effect = np.where(time <= 60, 0.3 * carbs, 0.1 * carbs)
    future_glucose = (
        glucose
        + carb_effect
        - 0.5 * np.sqrt(insulin) * insulin_sensitivity
        - 0.01 * time * glucose * insulin_sensitivity
    )
    future_glucose = np.clip(future_glucose, 0, 15)

    # Будущий инсулин: базальный уровень и реакция на повышение глюкозы
    basal_insulin = 5  # Базальный уровень инсулина
    insulin_response = 0.05 * np.maximum(glucose - 5, 0) * insulin_sensitivity
    future_insulin = (
        insulin
        + insulin_response
        - 0.005 * time * insulin
    )
    future_insulin = np.clip(future_insulin, 0, 50)

    # Если все входные значения равны 0, то выходные тоже равны 0
    mask = (glucose == 0) & (insulin == 0) & (carbs == 0) & (time == 0) & (time_of_day == 0)
    future_glucose[mask] = 0
    future_insulin[mask] = 0

    X = np.column_stack((glucose, insulin, carbs, time, time_of_day))
    y = np.column_stack((future_glucose, future_insulin))
    return X, y

# Генерация данных
X, y = generate_glucose_insulin_data()

# Определение архитектуры модели
model = Sequential([
    Dense(16, activation='relu', input_dim=5, kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(4, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(2)  # Выход: будущая глюкоза и будущий инсулин
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_mse)

# Обучение модели
model.fit(X, y, epochs=150, batch_size=64, validation_split=0.2, verbose=1)

# Сохранение модели
model.save("2complex_glucose_insulin_model_final.h5")
print("Модель сохранена как 2complex_glucose_insulin_model_final.h5")








































