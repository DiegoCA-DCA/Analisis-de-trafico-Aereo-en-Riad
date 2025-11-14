import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Cargo los datos
data = pd.read_parquet(r"C:\Users\LENOVO\Documents\Python\flights_RUH.parquet",engine="pyarrow")
#Hago una breve visualización de los datos
print(data.head(10))
#Analisis exploratorio de Datos
print(data.columns.tolist())
print(len(data.columns))
print(data.dtypes)
#Conversión de tipo de dato para las columnas a utilizar
data['airline.name'] = data['airline.name'].astype('category')
data['flight_type'] = data['flight_type'].astype('category')
data['movement.terminal'] = data['movement.terminal'].astype('category')
data['status'] = data['status'].astype('category')
#Convertimos las siguientes columnas a tipo Datetime
data['movement.scheduledTime.local'] = pd.to_datetime(data['movement.scheduledTime.local'], utc=True)
data['movement.scheduledTime.utc'] = pd.to_datetime(data['movement.scheduledTime.utc'], utc = True)
#Visulaización de datos Nulos por columna
print(data.isna().sum().sort_values(ascending=False))

# === 1. Top 8 Aerolíneas ===
FrecAero = (data.groupby('airline.name')['flight_number']
               .count()
               .sort_values(ascending=False)
               .head(8)
               .reset_index())
FrecAero.columns = ['Aerolinea', 'Vuelos']

# === 2. Vuelos por Status ===
DataStatus = data.groupby('status')['flight_number'].count().sort_values(ascending=False).reset_index()
DataStatus.columns = ['status', 'flight_number']

# === 3. Vuelos por hora (UTC) ===
vuelos_hora = data['movement.scheduledTime.utc'].dt.hour.value_counts().sort_index().reset_index()
vuelos_hora.columns = ['hora', 'vuelos']

# === 4. Serie temporal diaria ===
DataSerieTim = data.resample('D', on='movement.scheduledTime.local')['flight_number'].count().reset_index()
plt.style.use('seaborn-v0_8-whitegrid')  # Fondo limpio
fig = plt.figure(figsize=(18, 14), constrained_layout=True)  
# === SUBPLOTS con GridSpec para mejor control ===
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.1, wspace=0.1)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# === SUBPLOT 1: Top Aerolíneas ===
bars1 = ax1.bar(FrecAero['Aerolinea'], FrecAero['Vuelos'], 
                color='skyblue', edgecolor='navy', linewidth=1.2, alpha=0.9)
ax1.set_title('Top 8 Aerolíneas por Frecuencia', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Número de Vuelos', fontsize=12)
ax1.set_xlabel('')
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.grid(True, axis='y', alpha=0.4, linewidth=0.7)

# === SUBPLOT 2: Vuelos por Status ===
bars2 = ax2.bar(DataStatus['status'], DataStatus['flight_number'],
                color='lightgreen', edgecolor='darkgreen', linewidth=1.2, alpha=0.9)
ax2.set_title('Distribución por Estado del Vuelo', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Frecuencia', fontsize=12)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax2.tick_params(axis='x', rotation=0, labelsize=11)
ax2.grid(True, axis='y', alpha=0.4, linewidth=0.7)

# === SUBPLOT 3: Vuelos por Hora (UTC) ===
bars3 = ax3.bar(vuelos_hora['hora'], vuelos_hora['vuelos'],
                color='lightcoral', edgecolor='darkred', linewidth=1.2, alpha=0.9)
ax3.set_title('Distribución de Vuelos por Hora (UTC)', fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('Hora del Día (UTC)', fontsize=12)
ax3.set_ylabel('Número de Vuelos', fontsize=12)
max_y = vuelos_hora['vuelos'].max()
ax3.set_xticks(range(0, 24, 2))
ax3.grid(True, axis='y', alpha=0.4, linewidth=0.7)

# === SUBPLOT 4: Serie Temporal ===
line = ax4.plot(DataSerieTim['movement.scheduledTime.local'], DataSerieTim['flight_number'],
                color='red', linewidth=2.5)
ax4.set_title('Evolución Diaria de Vuelos', fontsize=14, fontweight='bold', pad=15)
ax4.set_xlabel('Fecha', fontsize=12)
ax4.set_ylabel('Número de Vuelos', fontsize=12)
ax4.grid(True, alpha=0.8, linewidth=0.8)
ax4.tick_params(axis='x', rotation=30, labelsize=10)

# === Fondo y bordes ===
for ax in [ax1, ax2, ax3, ax4]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

plt.show()