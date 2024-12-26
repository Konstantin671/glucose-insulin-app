import streamlit as st
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

# Кастомная функция потерь
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Кэшированная загрузка модели
@st.cache_resource
def load_model():
    try:
        model_path = "my_fixed_glucose_insulin_model.h5"
        model = tf.keras.models.load_model(model_path, custom_objects={"custom_mse": custom_mse})
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        st.stop()

def visualize_neural_network(weights, activations):
    try:
        G = nx.DiGraph()

        input_nodes = ['Глюкоза', 'Инсулин', 'Углеводы', 'Время', 'Время суток']
        hidden_layer_1_nodes = [f'H1_{i+1}' for i in range(weights[0].shape[1])]
        hidden_layer_2_nodes = [f'H2_{i+1}' for i in range(weights[1].shape[1])]
        hidden_layer_3_nodes = [f'H3_{i+1}' for i in range(weights[2].shape[1])]
        output_nodes = ['Будущая глюкоза', 'Будущий инсулин']

        G.add_nodes_from(input_nodes)
        G.add_nodes_from(hidden_layer_1_nodes)
        G.add_nodes_from(hidden_layer_2_nodes)
        G.add_nodes_from(hidden_layer_3_nodes)
        G.add_nodes_from(output_nodes)

        def add_edges(layer_weights, from_nodes, to_nodes, layer_activations):
            for i, from_node in enumerate(from_nodes):
                for j, to_node in enumerate(to_nodes):
                    weight = layer_weights[i, j]
                    activation = layer_activations[j]
                    intensity = (activation - np.min(layer_activations)) / (
                        np.max(layer_activations) - np.min(layer_activations) + 1e-8
                    )
                    color = plt.cm.RdYlBu(intensity)
                    G.add_edge(from_node, to_node, weight=weight, color=color)

        add_edges(weights[0], input_nodes, hidden_layer_1_nodes, activations[0])
        add_edges(weights[1], hidden_layer_1_nodes, hidden_layer_2_nodes, activations[1])
        add_edges(weights[2], hidden_layer_2_nodes, hidden_layer_3_nodes, activations[2])
        add_edges(weights[3], hidden_layer_3_nodes, output_nodes, activations[3])

        pos = {}
        for idx, node in enumerate(input_nodes):
            pos[node] = (0, idx * 5)
        for idx, node in enumerate(hidden_layer_1_nodes):
            pos[node] = (6, idx * 5)
        for idx, node in enumerate(hidden_layer_2_nodes):
            pos[node] = (12, idx * 5)
        for idx, node in enumerate(hidden_layer_3_nodes):
            pos[node] = (18, idx * 5)
        for idx, node in enumerate(output_nodes):
            pos[node] = (24, idx * 5)

        fig, ax = plt.subplots(figsize=(20, 14))
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        edge_colors = [d['color'] for _, _, d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, arrowstyle='-|>')

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Ошибка визуализации сети: {e}")

def main():
    st.title("Демонстрационная модель глюкозо-инсулиновой системы")

    model = load_model()

    st.write("""
В этом приложении вы можете ввести предполагаемые значения глюкозы, инсулина, 
количества потреблённых углеводов, времени до прогноза и времени суток. 
Модель покажет, как эти параметры могут повлиять на будущие уровни глюкозы и инсулина.
""")

    # Устанавливаем минимальные значения: глюкоза ≥ 2, инсулин ≥ 1, углеводы ≥ 5
    # Вы можете подправить верхние границы и default value под себя.
    glucose = st.slider("Глюкоза (ммоль/л)", 2.0, 12.0, value=5.0, step=0.1)
    insulin = st.slider("Инсулин (ед.)",     1.0, 30.0, value=8.0, step=0.5)
    carbs   = st.slider("Углеводы (граммы)", 5.0, 150.0, value=40.0, step=5.0)
    time    = st.slider("Время до прогноза (минуты)", 0, 180, value=30, step=5)

    time_of_day_label = st.selectbox("Время суток", ["Ночь (0)", "Утро (6)", "День (12)", "Вечер (18)"])
    time_of_day_map = {"Ночь (0)": 0, "Утро (6)": 6, "День (12)": 12, "Вечер (18)": 18}
    time_of_day = time_of_day_map[time_of_day_label]

    # Предупреждения при выборе минимальных значений
    if glucose == 2.0:
        st.warning("Вы выбрали минимально возможное значение глюкозы (2 ммоль/л). Это крайне низкая глюкоза!")
    if insulin == 1.0:
        st.warning("Вы выбрали минимальное значение инсулина (1 ед.). Это может означать почти полное отсутствие инсулина!")
    if carbs == 5.0:
        st.warning("Вы выбрали минимальное значение углеводов (5 г). Это очень мало углеводов для приёма!")

    # Формируем вход
    input_data = np.array([[glucose, insulin, carbs, time, time_of_day]], dtype=np.float32)

    # Предсказание
    try:
        output = model.predict(input_data)
        future_glucose  = output[0, 0]
        future_insulin  = output[0, 1]

        st.subheader("Результат предсказания:")
        st.write(f"**Будущая глюкоза**: {future_glucose:.2f} ммоль/л")
        st.write(f"**Будущий инсулин**: {future_insulin:.2f} ед.")

        # Визуализация сети
        weights = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights()) > 0]
        activations = []
        current_data = input_data
        for layer in model.layers:
            if len(layer.get_weights()) > 0:
                current_data = layer(current_data)
                activations.append(current_data.numpy().flatten())

        st.write("Визуализация нейронной сети:")
        visualize_neural_network(weights, activations)

    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")

if __name__ == "__main__":
    main()



# Кнопка для отображения описания
with st.expander("Подробнее об архитектуре модели и зависимостях"):
    st.markdown(r"""
    ## Подробное медицинское обоснование и описание модели

    ### Физиологические аспекты прогнозирования глюкозы и инсулина

    1. **Глюкоза (ммоль/л)**  
       - Является основным источником энергии для большинства тканей организма, особенно для головного мозга.  
       - Физиологический коридор глюкозы у здоровых людей: от 3.5 до 7.8 ммоль/л (натощак и в постпрандиальный период); у пациентов с сахарным диабетом данные показатели могут значительно меняться.  
       - Поддержание уровня гликемии зависит от баланса между поступлением глюкозы (из кишечника и/или из печени в виде гликогенолиза и глюконеогенеза) и её утилизацией (под влиянием инсулина и других гормонов).

    2. **Инсулин (единицы)**  
       - Гормон, синтезируемый бета-клетками поджелудочной железы, регулирует транспорт глюкозы из крови в клетки путём стимуляции GLUT4-переносчиков и подавления глюконеогенеза в печени.  
       - У пациентов с сахарным диабетом 1 типа инсулин вводится экзогенно (подкожно). При диабете 2 типа часто имеется инсулинорезистентность и/или относительный дефицит инсулина.  
       - Скорость деградации инсулина и период полужизни влияют на то, как быстро он «успевает» снижать уровень глюкозы.

    3. **Углеводы (граммы)**  
       - При приёме пищи углеводы гидролизуются до моносахаридов, главным образом глюкозы, которая всасывается в кровоток.  
       - Пик постпрандиальной гликемии (повышения уровня глюкозы после еды) зависит от гликемического индекса и состава пищи.  
       - Более быстро усваиваемые углеводы (например, сахароза) приводят к резкому скачку глюкозы, в то время как сложные углеводы повышают гликемию более плавно.

    4. **Время до прогноза (минуты, от 0 до 180)**  
       - В нашей задаче это горизонт прогноза, через который мы хотим узнать предполагаемые значения глюкозы и инсулина.  
       - За этот период могут успеть реализоваться различные процессы:
         - Абсорбция углеводов из ЖКТ,
         - Подъём эндогенного/экзогенного инсулина до «пика» действия,
         - Деградация инсулина и метаболизм глюкозы в тканях.

    5. **Время суток** (ночь, утро, день, вечер)  
       - Суточные ритмы (циркадные) влияют на гормональный фон. Например, утренние часы часто характеризуются феноменом «утренней зари», когда в результате ночного выброса контринсулярных гормонов (кортизол, гормон роста) повышается уровень глюкозы.  
       - Ночью чувствительность к инсулину может снижаться, а потребность в нём — варьироваться, что отражено в соответствующих коэффициентах модели.  
       - Учёт этих изменений помогает получить более точный прогноз, так как ответ на одну и ту же дозу инсулина/количество углеводов может существенно отличаться в разное время суток.

    ### Генерация синтетических данных с учётом физиологии
    - В процессе искусственной генерации данных заложены следующие ключевые принципы:
      - **Постепенное повышение глюкозы** после приёма углеводов (с ускоренным подъёмом при коротком промежутке времени и более плавным — при большем).
      - **Снижение уровня глюкозы** под воздействием инсулина, учитывая его циркадные особенности (утром чувствительность выше, ночью — ниже).  
      - **Метаболическая утилизация** (гликолиз, гликогеногенез) и эндогенная продукция глюкозы (глюконеогенез, гликогенолиз) в модели упрощённо отражены через формулу, учитывающую время и уровень инсулина.  
      - **Постепенная деградация инсулина** (примерно 50% за 4–5 минут для эндогенного, но мы используем обобщённую логику).

    ### Архитектура модели
    - **Полносвязная нейронная сеть** с тремя скрытыми слоями (16→8→4 нейронов) и релу-активацией.  
    - На вход подаются 5 признаков: глюкоза, инсулин, углеводы, время до прогноза и время суток.  
    - На выходе — два нейрона (будущий уровень глюкозы и инсулина).

    ### Процесс обучения
    1. **Данные**  
       - 10,000 сгенерированных примеров, в которых учтены базовые физиологические закономерности (действие инсулина и углеводов, циркадные ритмы).  
       - Целевые значения (глюкоза и инсулин через заданный интервал времени) рассчитываются исходя из упрощённой математической модели метаболизма.
    2. **Функция потерь**  
       - Кастомная среднеквадратичная ошибка (MSE), учитывающая разницу между предсказанной и истинной глюкозой/инсулином.
    3. **Оптимизатор**  
       - Adam (learning_rate=0.001), хорошо подходящий для обучения нейронных сетей с большим числом параметров.
    4. **Эпохи**  
       - 150 эпох, после которых достигается приемлемая сходимость ошибки на валидационной выборке.
    5. **Сохранение модели**  
       - Модель экспортируется в файл `my_fixed_glucose_insulin_model.h5` и далее используется в нашем Streamlit-приложении.

    ### Визуализация работы сети
    - Граф с узлами (нейронами) и рёбрами (весами):
      - Отображение текущей «активации» нейронов при введённых данных.
      - Толщина и цвет рёбер указывают на то, насколько сильно передаётся сигнал от одного узла к другому.
    - Такая визуализация позволяет понять, как сеть «решает» задачу, какие слои (нейроны) наиболее активно участвуют в формировании прогноза.

    ## Описание модели и процесса обучения

    ### Генерация данных
    Модель обучается на искусственно сгенерированных данных. Входные данные включают:
    1. **Текущий уровень глюкозы (ммоль/л)**.
    2. **Текущий уровень инсулина (единицы)**.
    3. **Количество потребленных углеводов (граммы)**.
    4. **Время до прогноза (минуты, от 0 до 180)**.
    5. **Время суток** (ночь, утро, день, вечер).

    Для генерации данных используются физиологически обоснованные зависимости:
    - Углеводы постепенно повышают уровень глюкозы.
    - Инсулин снижает уровень глюкозы, при этом его эффективность зависит от времени суток.
    - Со временем глюкоза перерабатывается, а инсулин деградирует.
    - Время суток влияет на чувствительность организма к инсулину: 
      - Утром — максимальная чувствительность.
      - Ночью — минимальная.

    ### Использование модели в Streamlit
    Приложение Streamlit:
    1. Загружает обученную модель.
    2. Предоставляет интерфейс с ползунками для ввода параметров:
       - Глюкоза, инсулин, углеводы, время до прогноза, время суток.
    3. Выполняет предсказание на основе введённых данных.
    4. Визуализирует структуру нейронной сети и активации.

    **Изменения в визуализации:**
    - Цвет рёбер между нейронами отображает уровень активации.
    - Толщина рёбер отражает силу соединений.

    #### Пример обработки слоя
    Вызов `add_edges(weights[0], input_nodes, hidden_layer_1_nodes, activations[0])` делает следующее:
    1. Извлекает веса соединений (weights[0]) между входными узлами (input_nodes) и узлами первого скрытого слоя (hidden_layer_1_nodes).
    2. Извлекает активации (activations[0]) узлов первого скрытого слоя.
    3. Добавляет рёбра между узлами, задавая:
       - Вес соединения.
       - Цвет рёбра в зависимости от активации узла назначения (hidden_layer_1_nodes).

    Каждый узел в скрытом слое — это один нейрон. Толщина, цвет или интенсивность окраски рёбер (линий между узлами) может соответствовать весу или уровню активации данного соединения:
    - Вес определяет, насколько сильно данный входной сигнал влияет на конкретный нейрон следующего слоя.
    - Оттенки (от синего к красному) связаны с уровнем активации или значением выхода нейрона. 

    Например, более «горячий» цвет (красный) может означать более высокую активацию данного нейрона при текущем наборе входных данных, а более «холодный» (синий) — более низкую активацию.

    ### Связь с результатами предсказаний
    Когда вы меняете входные параметры (глюкозу, инсулин, углеводы, время до прогноза, время суток) и нажимаете выполнить предсказание, модель пересчитывает значения по всем связям (рёбрам).
    Каждый нейрон в скрытых слоях получает взвешенную сумму входов и применяет к ней функцию активации (ReLU). 
    Получившиеся активации и веса определяют, какие узлы более или менее «активируются». 
    Наконец, выходные нейроны, на основе сигналов от последнего скрытого слоя, генерируют предсказанные значения будущей глюкозы и инсулина.

    Таким образом, цвет и интенсивность рёбер/узлов на графе во время визуализации показывают, как именно текущий набор входных данных «проходит» через сеть. 
    На практике, если вы измените входные данные и выполните предсказание повторно, расцветка рёбер и уровни активаций могут изменяться, отражая то, каким образом сеть перераспределяет свои внутренние сигналы, чтобы получить новые выходные результаты.

    Итог: На схеме вы видите архитектуру вашей обученной модели и то, как входные данные в данный момент времени активируют различные нейроны. 
    Эта визуализация даёт наглядное представление о том, как сеть «внутренне работает» при вычислении предсказаний будущих уровней глюкозы и инсулина.
                
    ### Заключение по визуализации
    В коде для визуализации используются:
    1. **Веса соединений (weights)** — для определения структуры графа (связей между узлами).
    2. **Уровни активации (activations)** — для задания цвета рёбер, что помогает понять, какие соединения \"активны\" при текущем наборе входных данных.
    Ключевое использование: Активации влияют на видимую интенсивность связи, а веса определяют, какие узлы соединены.

   ### Заключение
    1. **Модель** базируется на физиологически осмысленных принципах регуляции гликемии: углеводы повышают глюкозу, инсулин снижает, причём его эффективность различается в зависимости от времени суток.  
    2. **Нейронная сеть** с несколькими скрытыми слоями обучена на синтетических данных, отражающих действие инсулина и углеводов в краткосрочном прогнозе.  
    3. **Интерактивное приложение** даёт пользователю возможность увидеть будущий уровень глюкозы и инсулина при варьировании входных данных, а также наглядно наблюдать за тем, как внутренняя структура сети реагирует на изменения.
    """)

# Отдельная кнопка для отображения кода обучения модели
with st.expander("Показать код обучения модели"):
    st.code(r"""
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
# "больше глюкозы => в среднем больше инсулина"
# -----------------------------------------------------------------------------
def generate_glucose_insulin_data(samples=10000, random_seed=42):
    
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

    mask_A = (time <= 30)
    mask_B = (time > 30) & (time <= 60)
    mask_C = (time > 60)

    carb_effect = np.zeros(samples)
    carb_effect[mask_A] = 0.3 * carbs[mask_A]
    carb_effect[mask_B] = 0.2 * carbs[mask_B]
    carb_effect[mask_C] = 0.1 * carbs[mask_C]

    endo_production = np.zeros(samples)
    low_mask  = (glucose < 4)
    high_mask = (glucose > 8)
    mid_mask  = ~(low_mask | high_mask)

    endo_production[low_mask]  = np.random.uniform(0.8, 1.5, size=np.sum(low_mask))
    endo_production[high_mask] = np.random.uniform(0, 0.2,  size=np.sum(high_mask))
    endo_production[mid_mask]  = np.random.uniform(0, 0.5,  size=np.sum(mid_mask))


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
    future_glucose = np.clip(future_glucose, 0, 18) 

    base_secretion = np.zeros(samples)
    high_gluc_mask = (glucose > 6)

    base_secretion[high_gluc_mask] = 0.07 * (glucose[high_gluc_mask] - 6) * insulin_sensitivity[high_gluc_mask]

    fast_time = np.minimum(time, 30)
    slow_time = np.maximum(time - 30, 0)

    degr_fast = 0.005 * fast_time * insulin
    degr_slow = 0.002 * slow_time * insulin
    total_deg = degr_fast + degr_slow

    base_future_insulin = insulin + base_secretion - total_deg

    noise_ins = np.random.normal(0, 0.2, samples)
    future_insulin = base_future_insulin + noise_ins
    future_insulin = np.clip(future_insulin, 0, 40) 


    X = np.column_stack((glucose, insulin, carbs, time, time_of_day))
    y = np.column_stack((future_glucose, future_insulin))

    return X.astype(np.float32), y.astype(np.float32)

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
    print(f"\nМодель успешно сохранена в файл {MODEL_FILENAME}")""")

# Кнопка для отображения кода приложения Streamlit
with st.expander("Показать код приложения"):
    st.code(r"""import streamlit as st
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

# Кастомная функция потерь
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Кэшированная загрузка модели
@st.cache_resource
def load_model():
    try:
        model_path = "my_fixed_glucose_insulin_model.h5"
        model = tf.keras.models.load_model(model_path, custom_objects={"custom_mse": custom_mse})
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        st.stop()

def visualize_neural_network(weights, activations):
    try:
        G = nx.DiGraph()

        input_nodes = ['Глюкоза', 'Инсулин', 'Углеводы', 'Время', 'Время суток']
        hidden_layer_1_nodes = [f'H1_{i+1}' for i in range(weights[0].shape[1])]
        hidden_layer_2_nodes = [f'H2_{i+1}' for i in range(weights[1].shape[1])]
        hidden_layer_3_nodes = [f'H3_{i+1}' for i in range(weights[2].shape[1])]
        output_nodes = ['Будущая глюкоза', 'Будущий инсулин']

        G.add_nodes_from(input_nodes)
        G.add_nodes_from(hidden_layer_1_nodes)
        G.add_nodes_from(hidden_layer_2_nodes)
        G.add_nodes_from(hidden_layer_3_nodes)
        G.add_nodes_from(output_nodes)

        def add_edges(layer_weights, from_nodes, to_nodes, layer_activations):
            for i, from_node in enumerate(from_nodes):
                for j, to_node in enumerate(to_nodes):
                    weight = layer_weights[i, j]
                    activation = layer_activations[j]
                    intensity = (activation - np.min(layer_activations)) / (
                        np.max(layer_activations) - np.min(layer_activations) + 1e-8
                    )
                    color = plt.cm.RdYlBu(intensity)
                    G.add_edge(from_node, to_node, weight=weight, color=color)

        # Добавляем связи между слоями
        add_edges(weights[0], input_nodes, hidden_layer_1_nodes, activations[0])
        add_edges(weights[1], hidden_layer_1_nodes, hidden_layer_2_nodes, activations[1])
        add_edges(weights[2], hidden_layer_2_nodes, hidden_layer_3_nodes, activations[2])
        add_edges(weights[3], hidden_layer_3_nodes, output_nodes, activations[3])

        # Расположение узлов "лесенкой"
        pos = {}
        for idx, node in enumerate(input_nodes):
            pos[node] = (0, idx * 5)
        for idx, node in enumerate(hidden_layer_1_nodes):
            pos[node] = (6, idx * 5)
        for idx, node in enumerate(hidden_layer_2_nodes):
            pos[node] = (12, idx * 5)
        for idx, node in enumerate(hidden_layer_3_nodes):
            pos[node] = (18, idx * 5)
        for idx, node in enumerate(output_nodes):
            pos[node] = (24, idx * 5)

        fig, ax = plt.subplots(figsize=(20, 14))
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        edge_colors = [d['color'] for _, _, d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, arrowstyle='-|>')

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Ошибка визуализации сети: {e}")

def main():
    st.title("Демонстрационная модель глюкозо-инсулиновой системы")

    model = load_model()

    st.write(\"\"\"
В этом приложении вы можете ввести предполагаемые значения глюкозы, инсулина, 
количества потреблённых углеводов, времени до прогноза и времени суток. 
Модель покажет, как эти параметры могут повлиять на будущие уровни глюкозы и инсулина.
\"\"\")

    # Устанавливаем минимальные значения: глюкоза ≥ 2, инсулин ≥ 1, углеводы ≥ 5
    glucose = st.slider("Глюкоза (ммоль/л)", 2.0, 12.0, value=5.0, step=0.1)
    insulin = st.slider("Инсулин (ед.)", 1.0, 30.0, value=8.0, step=0.5)
    carbs = st.slider("Углеводы (граммы)", 5.0, 150.0, value=40.0, step=5.0)
    time = st.slider("Время до прогноза (минуты)", 0, 180, value=30, step=5)

    time_of_day_label = st.selectbox("Время суток", ["Ночь (0)", "Утро (6)", "День (12)", "Вечер (18)"])
    time_of_day_map = {"Ночь (0)": 0, "Утро (6)": 6, "День (12)": 12, "Вечер (18)": 18}
    time_of_day = time_of_day_map[time_of_day_label]

    # Предупреждения при выборе минимальных значений
    if glucose == 2.0:
        st.warning("Вы выбрали минимально возможное значение глюкозы (2 ммоль/л). Это крайне низкая глюкоза!")
    if insulin == 1.0:
        st.warning("Вы выбрали минимальное значение инсулина (1 ед.). Это может означать почти полное отсутствие инсулина!")
    if carbs == 5.0:
        st.warning("Вы выбрали минимальное значение углеводов (5 г). Это очень мало углеводов для приёма!")

    # Формируем входные данные для предсказания
    input_data = np.array([[glucose, insulin, carbs, time, time_of_day]], dtype=np.float32)

    # Предсказание
    try:
        output = model.predict(input_data)
        future_glucose = output[0, 0]
        future_insulin = output[0, 1]

        st.subheader("Результат предсказания:")
        st.write(f"**Будущая глюкоза**: {future_glucose:.2f} ммоль/л")
        st.write(f"**Будущий инсулин**: {future_insulin:.2f} ед.")

        st.write("Визуализация нейронной сети:")
        weights = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights()) > 0]
        activations = []
        current_data = input_data
        for layer in model.layers:
            if len(layer.get_weights()) > 0:
                current_data = layer(current_data)
                activations.append(current_data.numpy().flatten())

        visualize_neural_network(weights, activations)

    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")

if __name__ == "__main__":
    main()""")
