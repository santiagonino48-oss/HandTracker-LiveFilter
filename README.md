# Live Studio Hand Tracker & Filter 🖐️🎬

Filtro dinámico de seguimiento de manos en tiempo real desarrollado en **Python**, diseñado específicamente para streaming en vivo sin sobrecarga de recursos[cite: 1].

## ✨ Características Principales
- **Filtro B&W Dinámico:** Aplica un efecto de blanco y negro en el área encerrada entre ambas manos usando un polígono inteligente.
- **Activación por Gestos (5 Dedos):** Al abrir la palma de la mano completa, muestra de forma aleatoria una imagen flotante pequeña desde la carpeta de recursos.
- **Optimizado para Alto Rendimiento:** Utiliza MediaPipe Lite (`model_complexity=0`) y resolución fija a 640x480 para garantizar una transmisión fluida y sin tirones.
- **Integración con Unity Capture:** Envía la señal de video directo como cámara virtual limpia para software de streaming[cite: 1].
- **Interfaz Oculta Inteligente:** Los botones de control (Rotación y Modo Espejo) se ocultan automáticamente para que no interfieran con la captura de pantalla de tu directo.

---

## 🛠️ Tecnologías Utilizadas
- **Python** (Entorno virtual `venv`)
- **OpenCV** (`cv2`) para procesamiento de video y UI
- **MediaPipe** para el rastreo esquelético de manos
- **PyVirtualCam** y **Unity Capture** para la salida de cámara virtual[cite: 1]

---

## 🚀 Guía de Instalación y Uso

### 1. Requisitos previos
* Tener instalado **Python** (versión recomendada 3.10 o 3.11).
* Tener instalado el driver de **Unity Capture** para Windows (ejecutando su respectivo `.bat` como Administrador)[cite: 1].

### 2. Clonar el repositorio y configurar el entorno
Abre tu terminal en la carpeta del proyecto y crea un entorno virtual:
```bash
python -m venv venv
