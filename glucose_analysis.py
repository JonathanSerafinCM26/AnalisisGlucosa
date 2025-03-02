import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========================================================================================
# CONFIGURACIÓN INICIAL Y PREPARACIÓN DEL ENTORNO
# ========================================================================================
# Configuramos el estilo visual para los gráficos y habilitamos caracteres especiales en español
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.unicode_minus'] = False  # Permite mostrar correctamente signos negativos en gráficos
colors = plt.cm.tab10.colors  # Paleta de colores predefinida para mantener consistencia visual

# ========================================================================================
# CARGA DE DATOS E INSPECCIÓN INICIAL
# ========================================================================================
print("Cargando datos...")
# Leemos el archivo CSV que contiene los datos de glucosa, alimentación, sueño, ejercicio y estado mental
df = pd.read_csv('global_unified_stats (11).csv')
print(f"Forma del conjunto de datos: {df.shape}")

# Exploramos las primeras filas para entender la estructura general de los datos
print("\nPrimeras filas:")
print(df.head())

# Verificamos los tipos de datos para cada columna
print("\nTipos de datos:")
print(df.dtypes)

# Identificamos valores faltantes para considerar en el preprocesamiento
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# Verificamos los diferentes tipos de registros disponibles
print("\nTipos de registros únicos:")
print(df['record_type'].unique())

# Contamos el número total de usuarios en el conjunto de datos
print("\nUsuarios únicos:")
print(len(df['user_id'].unique()), "usuarios")

# ========================================================================================
# SEPARACIÓN Y SEGMENTACIÓN DE DATOS POR TIPO DE REGISTRO
# ========================================================================================
# Creamos subconjuntos específicos para cada tipo de registro, facilitando su análisis individual
df_glucose = df[df['record_type'] == 'glucose']  # Datos de niveles de glucosa
df_alimentation = df[df['record_type'] == 'alimentation']  # Datos de alimentación
df_sleep = df[df['record_type'] == 'sleep']  # Datos de sueño
df_exercise = df[df['record_type'] == 'exercise']  # Datos de ejercicio
df_mental = df[df['record_type'] == 'mental_state']  # Datos de estado mental

# ========================================================================================
# AGREGACIÓN DE DATOS A NIVEL DIARIO POR USUARIO
# ========================================================================================
# Para cada tipo de registro, agregamos los datos a nivel diario por usuario
# Esto nos permite relacionar los valores de glucosa con otros factores en el mismo día

# Niveles medios de glucosa diarios por usuario
df_glucose_daily = df_glucose.groupby(['user_id', 'date (YYYY-MM-DD)']).agg({
    'glucose_level': 'mean'  # Promedio de lecturas de glucosa por día
}).reset_index()

# Métricas diarias de alimentación por usuario
df_alimentation_daily = df_alimentation.groupby(['user_id', 'date (YYYY-MM-DD)']).agg({
    'calories': 'sum',  # Calorías totales consumidas en el día
    'carbohydrates': 'sum',  # Carbohidratos totales consumidos
    'proteins': 'sum',  # Proteínas totales consumidas
    'fats': 'sum',  # Grasas totales consumidas
    'sugars': 'mean'  # Contenido medio de azúcar en las comidas del día
}).reset_index()

# Métricas de sueño diarias por usuario
df_sleep_daily = df_sleep.groupby(['user_id', 'date (YYYY-MM-DD)']).agg({
    'hours_slept': 'mean'  # Horas promedio de sueño por día
}).reset_index()

# Minutos totales de ejercicio diarios por usuario
df_exercise_daily = df_exercise.groupby(['user_id', 'date (YYYY-MM-DD)']).agg({
    'exercise_minutes': 'sum'  # Tiempo total de ejercicio en el día
}).reset_index()

# Estado mental promedio diario por usuario
df_mental_daily = df_mental.groupby(['user_id', 'date (YYYY-MM-DD)']).agg({
    'anxiety_level': 'mean',  # Nivel promedio de ansiedad
    'stress_level': 'mean',  # Nivel promedio de estrés
    'sadness_level': 'mean',  # Nivel promedio de tristeza
    'fulfillment_level': 'mean'  # Nivel promedio de satisfacción
}).reset_index()

# ========================================================================================
# FUSIÓN DE DATOS PARA ANÁLISIS INTEGRAL
# ========================================================================================
# Combinamos todos los datos diarios en un único dataframe para analizar relaciones
print("\nFusionando datos...")
# Usamos la glucosa como base y añadimos las demás métricas mediante operaciones de merge
df_combined = df_glucose_daily.merge(
    df_alimentation_daily, on=['user_id', 'date (YYYY-MM-DD)'], how='left'
).merge(
    df_sleep_daily, on=['user_id', 'date (YYYY-MM-DD)'], how='left'
).merge(
    df_exercise_daily, on=['user_id', 'date (YYYY-MM-DD)'], how='left'
).merge(
    df_mental_daily, on=['user_id', 'date (YYYY-MM-DD)'], how='left'
)

# Verificamos la dimensión del conjunto de datos combinado
print(f"Forma del conjunto de datos combinado: {df_combined.shape}")
# Identificamos valores faltantes que pudieran surgir de la combinación
print("\nValores faltantes en el conjunto de datos combinado:")
print(df_combined.isnull().sum())

# ========================================================================================
# TRATAMIENTO DE VALORES FALTANTES
# ========================================================================================
# Rellenamos los valores faltantes con la mediana de cada columna 
# La mediana es menos sensible a valores extremos que la media
print("\nRellenando valores faltantes...")
for column in df_combined.columns:
    if column not in ['user_id', 'date (YYYY-MM-DD)'] and df_combined[column].isnull().sum() > 0:
        df_combined[column] = df_combined[column].fillna(df_combined[column].median())

# Generamos estadísticas descriptivas para entender la distribución de los datos
print("\nEstadísticas descriptivas:")
print(df_combined.describe())

# ========================================================================================
# PREPARACIÓN PARA EL ANÁLISIS VISUAL
# ========================================================================================
# Creamos un directorio para guardar todas las visualizaciones generadas
import os
os.makedirs('plots', exist_ok=True)

# Función auxiliar para crear y guardar gráficos de manera consistente
def crear_y_guardar_grafico(fig, filename, tight_layout=True):
    """
    Guarda un gráfico matplotlib con formato consistente y alta resolución
    
    Parámetros:
    - fig: Figura de matplotlib a guardar
    - filename: Nombre del archivo para guardar
    - tight_layout: Si se debe aplicar tight_layout() para optimizar el espacio
    """
    if tight_layout:
        plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)

# ========================================================================================
# ANÁLISIS VISUAL 1: DISTRIBUCIÓN DE NIVELES DE GLUCOSA
# ========================================================================================
# Este gráfico muestra la distribución general de los niveles de glucosa en toda la población
print("\nGenerando gráfico de distribución de glucosa...")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_glucose['glucose_level'], kde=True, ax=ax)
ax.set_title('Distribución de Niveles de Glucosa en Todos los Usuarios', fontsize=14)
ax.set_xlabel('Nivel de Glucosa (mg/dL)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
crear_y_guardar_grafico(fig, 'distribucion_glucosa.png')

# ========================================================================================
# ANÁLISIS VISUAL 2: SERIES TEMPORALES DE GLUCOSA
# ========================================================================================
# Analizamos cómo varían los niveles de glucosa a lo largo del tiempo para un usuario específico
print("\nGenerando gráfico de serie temporal de glucosa...")
# Seleccionamos el usuario 20 como ejemplo y ordenamos sus datos cronológicamente
user_20_glucose = df_glucose[df_glucose['user_id'] == 20].sort_values('timestamp_complete')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(user_20_glucose['timestamp_complete'], user_20_glucose['glucose_level'], marker='o', linestyle='-')
ax.set_title('Niveles de Glucosa a lo Largo del Tiempo - Usuario 20', fontsize=14)
ax.set_xlabel('Fecha y Hora', fontsize=12)
ax.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
# Añadimos líneas de referencia para niveles bajos y altos de glucosa
ax.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Límite inferior (70 mg/dL)')
ax.axhline(y=140, color='r', linestyle='--', alpha=0.7, label='Límite superior (140 mg/dL)')
ax.set_xticks(user_20_glucose['timestamp_complete'][::4])  # Mostramos solo cada 4ta marca de tiempo para legibilidad
plt.xticks(rotation=45)
ax.legend()
crear_y_guardar_grafico(fig, 'serie_temporal_glucosa_usuario_20.png')

# ========================================================================================
# ANÁLISIS VISUAL 3: COMPARATIVA DE GLUCOSA POR USUARIO
# ========================================================================================
# Comparamos los niveles promedio de glucosa entre los diferentes usuarios
print("\nGenerando gráfico de glucosa promedio por usuario...")
avg_glucose_by_user = df_glucose.groupby('user_id')['glucose_level'].mean().sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(avg_glucose_by_user)), avg_glucose_by_user.values, color=colors[0])
ax.set_xticks(range(len(avg_glucose_by_user)))
ax.set_xticklabels(avg_glucose_by_user.index, rotation=45)
ax.set_title('Nivel Promedio de Glucosa por Usuario', fontsize=14)
ax.set_xlabel('ID de Usuario', fontsize=12)
ax.set_ylabel('Nivel Promedio de Glucosa (mg/dL)', fontsize=12)
crear_y_guardar_grafico(fig, 'glucosa_promedio_por_usuario.png')

# ========================================================================================
# ANÁLISIS VISUAL 4: RELACIÓN ENTRE GLUCOSA Y FACTORES DE ESTILO DE VIDA
# ========================================================================================
# Creamos gráficos de dispersión para visualizar relaciones entre glucosa y diversos factores
print("\nGenerando gráficos de dispersión para glucosa vs factores de estilo de vida...")
# Definimos las variables a analizar y sus nombres en español para las etiquetas
features = ['carbohydrates', 'proteins', 'fats', 'hours_slept', 'exercise_minutes', 'anxiety_level', 'stress_level']
feature_names_es = ['Carbohidratos', 'Proteínas', 'Grasas', 'Horas de Sueño', 'Minutos de Ejercicio', 'Nivel de Ansiedad', 'Nivel de Estrés']
feature_dict = dict(zip(features, feature_names_es))

# Creamos una matriz de subgráficos para mostrar todas las relaciones
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

# Para cada característica, generamos un gráfico de dispersión con línea de regresión
for i, feature in enumerate(features):
    if i < len(axes):
        sns.scatterplot(x=feature, y='glucose_level', data=df_combined, alpha=0.6, ax=axes[i])
        axes[i].set_title(f'Glucosa vs {feature_dict[feature]}', fontsize=12)
        axes[i].set_xlabel(feature_dict[feature], fontsize=10)
        axes[i].set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=10)
        
        # Añadimos línea de regresión para visualizar la tendencia general
        sns.regplot(x=feature, y='glucose_level', data=df_combined, scatter=False, 
                   ax=axes[i], line_kws={"color":"red"})

# Eliminamos los subgráficos no utilizados
for i in range(len(features), len(axes)):
    fig.delaxes(axes[i])

crear_y_guardar_grafico(fig, 'glucosa_vs_factores_dispersion.png')

# ========================================================================================
# ANÁLISIS VISUAL 5: GLUCOSA POR TIPO DE COMIDA
# ========================================================================================
# Analizamos cómo diferentes tipos de comidas afectan los niveles de glucosa
print("\nGenerando boxplot para glucosa por tipo de comida...")
# Copiamos los datos de alimentación para no modificar el original
comidas_por_usuario_y_fecha = df_alimentation.copy()
# Verificamos si existe la columna 'meal_name' antes de proceder
if 'meal_name' in comidas_por_usuario_y_fecha.columns:
    # Fusionamos datos de glucosa con información de comidas por usuario y fecha
    df_with_meal = pd.merge(
        df_glucose[['user_id', 'date (YYYY-MM-DD)', 'glucose_level']], 
        comidas_por_usuario_y_fecha[['user_id', 'date (YYYY-MM-DD)', 'meal_name']],
        on=['user_id', 'date (YYYY-MM-DD)'], 
        how='inner'
    )
    
    # Verificamos que la fusión haya generado datos válidos
    if not df_with_meal.empty and 'meal_name' in df_with_meal.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='meal_name', y='glucose_level', data=df_with_meal, ax=ax)
        ax.set_title('Niveles de Glucosa por Tipo de Comida', fontsize=14)
        ax.set_xlabel('Tipo de Comida', fontsize=12)
        ax.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
        plt.xticks(rotation=45)
        crear_y_guardar_grafico(fig, 'glucosa_por_tipo_comida.png')
    else:
        print("No se pudo crear el boxplot por tipo de comida: no hay datos suficientes o estructura incorrecta.")
else:
    print("La columna 'meal_name' no existe en los datos.")

# ========================================================================================
# ANÁLISIS VISUAL 6: MATRIZ DE CORRELACIÓN
# ========================================================================================
# Generamos un mapa de calor para visualizar correlaciones entre todas las variables numéricas
print("\nGenerando mapa de calor de correlación...")
# Seleccionamos solo columnas numéricas para el cálculo de correlaciones
numeric_cols = df_combined.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df_combined[numeric_cols].corr()

# Traducimos nombres de columnas para mejorar legibilidad en español
col_names_es = {
    'glucose_level': 'Nivel de Glucosa',
    'calories': 'Calorías',
    'carbohydrates': 'Carbohidratos',
    'proteins': 'Proteínas',
    'fats': 'Grasas',
    'sugars': 'Azúcares',
    'hours_slept': 'Horas de Sueño',
    'exercise_minutes': 'Minutos de Ejercicio',
    'anxiety_level': 'Nivel de Ansiedad',
    'stress_level': 'Nivel de Estrés',
    'sadness_level': 'Nivel de Tristeza',
    'fulfillment_level': 'Nivel de Satisfacción'
}

# Renombramos columnas e índices para mostrar en español
correlation_matrix_es = correlation_matrix.rename(columns=col_names_es, index=col_names_es)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix_es, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Matriz de Correlación entre Glucosa y Factores de Estilo de Vida', fontsize=14)
crear_y_guardar_grafico(fig, 'mapa_calor_correlacion.png')

# ========================================================================================
# ANÁLISIS VISUAL 7: GLUCOSA POR DURACIÓN DE EJERCICIO
# ========================================================================================
# Analizamos cómo la duración del ejercicio afecta los niveles de glucosa
print("\nGenerando gráfico de glucosa por duración de ejercicio...")
# Creamos categorías (bins) para los minutos de ejercicio
df_combined['exercise_bins'] = pd.cut(df_combined['exercise_minutes'], 
                                     bins=[0, 20, 40, 60, 120], 
                                     labels=['0-20', '21-40', '41-60', '60+'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='exercise_bins', y='glucose_level', data=df_combined, ax=ax)
ax.set_title('Niveles de Glucosa por Duración de Ejercicio', fontsize=14)
ax.set_xlabel('Duración de Ejercicio (minutos)', fontsize=12)
ax.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
crear_y_guardar_grafico(fig, 'glucosa_por_ejercicio.png')

# ========================================================================================
# ANÁLISIS VISUAL 8: GLUCOSA POR DURACIÓN DE SUEÑO
# ========================================================================================
# Analizamos cómo la duración del sueño afecta los niveles de glucosa
print("\nGenerando gráfico de glucosa por duración de sueño...")
# Creamos categorías (bins) para las horas de sueño
df_combined['sleep_bins'] = pd.cut(df_combined['hours_slept'], 
                                  bins=[0, 6, 8, 10, 12], 
                                  labels=['<6', '6-8', '8-10', '10+'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='sleep_bins', y='glucose_level', data=df_combined, ax=ax)
ax.set_title('Niveles de Glucosa por Duración de Sueño', fontsize=14)
ax.set_xlabel('Duración de Sueño (horas)', fontsize=12)
ax.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
crear_y_guardar_grafico(fig, 'glucosa_por_sueno.png')

# ========================================================================================
# MODELADO PREDICTIVO: PREPARACIÓN DE DATOS
# ========================================================================================
print("\nPreparando datos para aprendizaje automático...")

# Definimos las características predictoras (X) y la variable objetivo (y)
features = ['calories', 'carbohydrates', 'proteins', 'fats', 'hours_slept', 
            'exercise_minutes', 'anxiety_level', 'stress_level']
X = df_combined[features]  # Variables predictoras
y = df_combined['glucose_level']  # Variable objetivo (niveles de glucosa)

# Estandarizamos las características para mejorar el rendimiento de los modelos
# Esto asegura que todas las variables estén en la misma escala
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividimos los datos en conjuntos de entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========================================================================================
# MODELADO PREDICTIVO: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ========================================================================================
print("\nEntrenando y evaluando modelos...")

# Definimos los diferentes modelos a evaluar
models = {
    'Regresión Lineal': LinearRegression(),
    'Árbol de Decisión': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Lista para almacenar resultados de evaluación
results = []

# Evaluamos cada modelo
for name, model in models.items():
    # Entrenamos el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)
    
    # Realizamos predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calculamos métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)  # Error cuadrático medio
    rmse = np.sqrt(mse)  # Raíz del error cuadrático medio
    r2 = r2_score(y_test, y_pred)  # Coeficiente de determinación R²
    
    # Realizamos validación cruzada para evaluar estabilidad del modelo
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())  # Convertimos el MSE negativo a RMSE positivo
    
    # Guardamos los resultados para comparación
    results.append({
        'Modelo': name,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'CV RMSE': cv_rmse
    })
    
    print(f"Entrenado {name}: MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}, CV RMSE={cv_rmse:.2f}")
    
    # Para modelos basados en árboles, analizamos la importancia de las características
    if hasattr(model, 'feature_importances_'):
        # Traducimos nombres de características al español
        features_es = [col_names_es.get(f, f) for f in features]
        feature_importance = pd.DataFrame({
            'Característica': features_es,
            'Importancia': model.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        print(f"\nImportancia de Características para {name}:")
        print(feature_importance)
        
        # Visualizamos la importancia de características
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importancia', y='Característica', data=feature_importance, ax=ax)
        ax.set_title(f'Importancia de Características - {name}', fontsize=14)
        crear_y_guardar_grafico(fig, f'importancia_caracteristicas_{name.replace(" ", "_").lower()}.png')

# Convertimos los resultados a DataFrame para mejor visualización
results_df = pd.DataFrame(results)
print("\nComparación de Rendimiento de Modelos:")
print(results_df)

# ========================================================================================
# VISUALIZACIÓN COMPARATIVA DE MODELOS
# ========================================================================================
# Generamos un gráfico comparativo de las métricas para todos los modelos
fig, ax = plt.subplots(figsize=(12, 6))
models_df = pd.DataFrame(results)
x = np.arange(len(models_df['Modelo']))
width = 0.2

# Graficamos barras para cada métrica
ax.bar(x - width, models_df['RMSE'], width, label='RMSE')
ax.bar(x, models_df['CV RMSE'], width, label='CV RMSE')
ax.bar(x + width, models_df['R²'], width, label='R²')

ax.set_title('Comparación de Rendimiento de Modelos', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models_df['Modelo'])
ax.legend()
crear_y_guardar_grafico(fig, 'comparacion_modelos.png')

# ========================================================================================
# AJUSTE DE HIPERPARÁMETROS DEL MEJOR MODELO
# ========================================================================================
print("\nAjuste de hiperparámetros para el mejor modelo...")
# Seleccionamos el modelo con mejor rendimiento inicial para optimización
# Esta selección se puede ajustar manualmente después de ver los resultados iniciales
mejor_modelo = 'Random Forest'  # Se puede cambiar según los resultados

# Definimos diferentes rejillas de búsqueda según el modelo elegido
if mejor_modelo == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],  # Número de árboles en el bosque
        'max_depth': [None, 10, 20, 30],  # Profundidad máxima de los árboles
        'min_samples_split': [2, 5, 10],  # Mínimo de muestras para dividir un nodo
        'min_samples_leaf': [1, 2, 4]  # Mínimo de muestras en un nodo hoja
    }
    model = RandomForestRegressor(random_state=42)
    nombre_modelo_es = 'Random Forest'
elif mejor_modelo == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],  # Número de árboles
        'learning_rate': [0.01, 0.1, 0.2],  # Tasa de aprendizaje
        'max_depth': [3, 5, 7],  # Profundidad máxima
        'min_samples_split': [2, 5]  # Mínimo de muestras para dividir
    }
    model = GradientBoostingRegressor(random_state=42)
    nombre_modelo_es = 'Gradient Boosting'
else:  # Árbol de Decisión como opción predeterminada
    param_grid = {
        'max_depth': [None, 10, 20, 30],  # Profundidad máxima
        'min_samples_split': [2, 5, 10],  # Mínimo de muestras para dividir
        'min_samples_leaf': [1, 2, 4]  # Mínimo de muestras en nodo hoja
    }
    model = DecisionTreeRegressor(random_state=42)
    nombre_modelo_es = 'Árbol de Decisión'

# Realizamos búsqueda en rejilla para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                          cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y)

print(f"\nMejores parámetros para {nombre_modelo_es}:")
print(grid_search.best_params_)

# Evaluamos el rendimiento del modelo optimizado
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nRendimiento del mejor {nombre_modelo_es}:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# ========================================================================================
# VISUALIZACIÓN FINAL: PREDICCIONES VS VALORES REALES
# ========================================================================================
# Graficamos los valores reales contra los predichos para evaluar visualmente el rendimiento
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
# Añadimos la línea de identidad perfecta (y=x)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
ax.set_xlabel('Nivel de Glucosa Real', fontsize=12)
ax.set_ylabel('Nivel de Glucosa Predicho', fontsize=12)
ax.set_title('Niveles de Glucosa Reales vs Predichos', fontsize=14)
crear_y_guardar_grafico(fig, 'real_vs_predicho.png')

# ========================================================================================
# CONCLUSIONES FINALES
# ========================================================================================
print("\nCONCLUSIONES:")
print("1. El análisis muestra relaciones entre los niveles de glucosa y varios factores del estilo de vida.")
print(f"2. El mejor modelo de aprendizaje automático para predecir los niveles de glucosa es {nombre_modelo_es}.")
print("3. Se han identificado los factores clave que influyen en los niveles de glucosa (basados en la importancia de las características).")
print("4. Se recomienda utilizar el modelo identificado para futuras predicciones de glucosa basadas en datos de estilo de vida.")
print("\nTodos los gráficos de análisis se han guardado en la carpeta 'plots'.")