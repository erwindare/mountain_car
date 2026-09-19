# Consideraciones a tener en cuenta para la ejecución de los agentes

Los siguientes comandos deben ser ejecutados estando posicionado en la carpeta **mountain_car**.

Primero verificar si existen los archivos binarios de los agentes con el siguiente comando: **uv run mountaincar list**

En caso que existan, eliminarlos así: **uv run mountaincar delete <nombre_agente>**

## Ejecución del agente Q-Learning:

Creación de agente nuevo para Q-Learning: **uv run mountaincar init qlearning**

Entrenamiento del agente: **uv run mountaincar train qlearning --episodes 20000**

Evaluación del agente: **uv run mountaincar load qlearning --eval**

Verificación de la curva de aprendizaje del agente: **.\.venv\Scripts\python.exe .\graficos\graficas_qlearning.py --episodes 20000 --window 100**

Verificación del comportamiento del agente en una ventana de windows: **uv run mountaincar render qlearning --episodes 3**


## Ejecución del agente DQN:

Creación de agente nuevo para DQN: **uv run mountaincar init dqn**

Entrenamiento del agente: **uv run mountaincar train dqn --episodes 2500**

Evaluación del agente: **uv run mountaincar load dqn --eval**

Verificación de la curva de aprendizaje del agente: **.\.venv\Scripts\python.exe .\graficos\graficas_dqn.py --episodes 2500 --window 100**

Verificación del comportamiento del agente en una ventana de windows: **uv run mountaincar render dqn --episodes 3**
