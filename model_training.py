import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# -----------------------------------------------------------------------------
# Кастомная функция ошибки (MSE)
# -----------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# -----------------------------------------------------------------------------
# Улучшенная генерация данных с более корректной зависимостью 
# "больше глюкозы => в среднем больше инсулина"
# -----------------------------------------------------------------------------
def generate_glucose_insulin_data(samples=10000, random_seed=42):
    """
    Генерация синтетических данных с правками для более физиологичной связи:
      - Если глюкоза выше, как правило, инсулин тоже в итоге выше (за счёт секреции).
      - Деградация инсулина ослаблена, чтобы секреция при высокой глюкозе 
        перекрывала «утечку».
      - Остальные факторы (углеводы, время, циркадный ритм) сохранены.
    """
    np.random.seed(random_seed)

    # ---------------------------
    # 1) Генерация исходных данных
    # ---------------------------
    glucose = np.random.uniform(4, 9, samples)       # Текущая глюкоза, ммоль/л (4..9)
    insulin = np.random.uniform(5, 15, samples)      # Текущий инсулин, ед. (5..15)
    carbs   = np.random.uniform(0, 100, samples)     # Углеводы, граммы (0..100)
    time    = np.random.uniform(0, 120, samples)     # Горизонт прогноза, мин (0..120)
    time_of_day = np.random.choice([0, 6, 12, 18], samples)

    # Циркадная чувствительность
    insulin_sensitivity_map = {
        0: 0.7,   # Ночь
        6: 1.2,   # Утро
        12: 1.0,  # День
        18: 0.9   # Вечер
    }
    insulin_sensitivity = np.array([insulin_sensitivity_map[t] for t in time_of_day])

    # ---------------------------
    # 2) Модель будущей глюкозы
    # ---------------------------
    # Упрощённый постпрандиальный подъём: три зоны
    #   0..30  => +0.3 * carbs
    #   30..60 => +0.2 * carbs
    #   >60    => +0.1 * carbs
    mask_A = (time <= 30)
    mask_B = (time > 30) & (time <= 60)
    mask_C = (time > 60)

    carb_effect = np.zeros(samples)
    carb_effect[mask_A] = 0.3 * carbs[mask_A]
    carb_effect[mask_B] = 0.2 * carbs[mask_B]
    carb_effect[mask_C] = 0.1 * carbs[mask_C]

    # Эндогенная продукция: чуть снижаем, чтобы глюкоза не «улетала» 
    # при и так высоком уровне
    endo_production = np.zeros(samples)
    low_mask  = (glucose < 4)
    high_mask = (glucose > 8)
    mid_mask  = ~(low_mask | high_mask)

    endo_production[low_mask]  = np.random.uniform(0.8, 1.5, size=np.sum(low_mask))
    endo_production[high_mask] = np.random.uniform(0, 0.2,  size=np.sum(high_mask))
    endo_production[mid_mask]  = np.random.uniform(0, 0.5,  size=np.sum(mid_mask))

    # Усилим влияние инсулина на снижение глюкозы, чтобы связь была заметна
    base_future_glucose = (
        glucose
        + carb_effect
        - 0.7 * np.sqrt(insulin) * insulin_sensitivity
        - 0.005 * time * glucose * insulin_sensitivity
        + endo_production
    )

    # Умеренный шум (±0.15)
    noise_g = np.random.normal(0, 0.15, samples)
    future_glucose = base_future_glucose + noise_g
    future_glucose = np.clip(future_glucose, 0, 18)  # обрежем диапазон 0..18

    # ---------------------------
    # 3) Модель будущего инсулина
    # ---------------------------
    # Усилим секрецию при глюкозе>6, чтобы при росте глюкозы инсулин 
    # однозначно вырастал (если time не слишком велик).
    # Ослабим деградацию, чтобы её не хватало «убить» инсулин при высоких уровнях глюкозы.
    base_secretion = np.zeros(samples)
    high_gluc_mask = (glucose > 6)
    # Секреция сильнее (0.07 вместо 0.04/0.05)
    base_secretion[high_gluc_mask] = 0.07 * (glucose[high_gluc_mask] - 6) * insulin_sensitivity[high_gluc_mask]

    # Деградация: двухфазная, но со сниженным коэффициентом
    #   fast: 0.005 * min(time, 30)
    #   slow: 0.002 * max(time-30, 0)
    fast_time = np.minimum(time, 30)
    slow_time = np.maximum(time - 30, 0)

    degr_fast = 0.005 * fast_time * insulin
    degr_slow = 0.002 * slow_time * insulin
    total_deg = degr_fast + degr_slow

    base_future_insulin = insulin + base_secretion - total_deg

    # Уменьшенный шум (±0.2)
    noise_ins = np.random.normal(0, 0.2, samples)
    future_insulin = base_future_insulin + noise_ins
    future_insulin = np.clip(future_insulin, 0, 40)  # Диапазон 0..40

    # ---------------------------
    # Собираем X, y
    # ---------------------------
    X = np.column_stack((glucose, insulin, carbs, time, time_of_day))
    y = np.column_stack((future_glucose, future_insulin))

    return X.astype(np.float32), y.astype(np.float32)

# -----------------------------------------------------------------------------
# Основная часть: обучение модели
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Генерируем данные
    X, y = generate_glucose_insulin_data(samples=10000, random_seed=42)
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    # 2) Определяем модель
    model = Sequential([
        Dense(16, activation='relu', input_dim=5, kernel_regularizer=l2(0.001), name="dense_1"),
        Dropout(0.3, name="dropout_1"),
        Dense(8, activation='relu', kernel_regularizer=l2(0.001), name="dense_2"),
        Dropout(0.3, name="dropout_2"),
        Dense(4, activation='relu', kernel_regularizer=l2(0.001), name="dense_3"),
        Dropout(0.3, name="dropout_3"),
        Dense(2, name="output_layer")
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_mse)

    # 3) Обучаем (150 эпох)
    model.fit(X, y, epochs=150, batch_size=64, validation_split=0.2, verbose=1)

    # 4) Сохраняем
    MODEL_FILENAME = "my_fixed_glucose_insulin_model.h5"
    model.save(MODEL_FILENAME)
    print(f"\nМодель успешно сохранена в файл {MODEL_FILENAME}")



















































