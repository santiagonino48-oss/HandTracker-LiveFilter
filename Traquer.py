import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam
import os
import math
import random
import time
import threading

# Carpeta donde está este script (así "img" se encuentra aunque lo lances desde otra ruta)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VENTANA = "Hand Tracker - Frame"
VENTANA_DEBUG = "Hand Tracker - Con Trackers (Debug)"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Modelo con más precisión para dedos vistos de frente
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,  # 1 = más preciso con dedos de frente/tapados (pon 0 si notas tirones)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ----------------------------------------------------------------------
# CÁMARA EN HILO APARTE (evita retrasos y tirones al leer frames)
# ----------------------------------------------------------------------
class CamaraHilo:
    def __init__(self, indice=0, ancho=640, alto=480):
        self.cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(indice)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        self.frame = None
        self.lock = threading.Lock()
        self.activo = True
        self.hilo = threading.Thread(target=self._leer, daemon=True)

    def abierta(self):
        return self.cap.isOpened()

    def iniciar(self):
        self.hilo.start()
        return self

    def _leer(self):
        while self.activo:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def leer(self):
        with self.lock:
            return self.frame

    def detener(self):
        self.activo = False
        self.hilo.join(timeout=1)
        self.cap.release()


camara = CamaraHilo(0, 640, 480)
if not camara.abierta():
    raise SystemExit("No se pudo abrir la cámara (índice 0). Ciérrala en otras apps e inténtalo de nuevo.")

width = int(camara.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(camara.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = camara.cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30

camara.iniciar()

# ----------------------------------------------------------------------
# ESTADO
# ----------------------------------------------------------------------
rotation_state = 0
mirror_mode = True  # Estado inicial del modo espejo (Activado)
show_debug_window = False

# Delay de activación de las 2 manos (1 segundo) y tolerancia a pérdidas breves de detección
delay_required = 1.0
GRACIA_MANOS = 0.3
two_hands_timer = None
ultimo_visto_dos = 0.0
effect_active = False

# Tolerancia para que la imagen no parpadee si la mano falla un instante
GRACIA_IMG = 0.4
t_ultima_abierta = 0.0

# Coordenadas de los botones arriba
btn_left = [20, 20, 110, 60]
btn_right = [120, 20, 210, 60]
btn_mirror = [220, 20, 310, 60]
btn_tracker = [320, 20, 430, 60]

show_buttons = False


# ----------------------------------------------------------------------
# IMÁGENES (carpeta "img": mybad 1, mybad 2, mybad 3)
# ----------------------------------------------------------------------
def a_bgra(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def cargar_imagenes():
    carpetas = [os.path.join(BASE_DIR, "img"), "img"]
    imgs = []
    for i in range(1, 4):
        encontrada = False
        for carpeta in carpetas:
            for ext in ('.png', '.jpg', '.jpeg'):
                for nombre in (f"mybad {i}{ext}", f"mybad{i}{ext}"):
                    ruta = os.path.join(carpeta, nombre)
                    if os.path.exists(ruta):
                        img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            imgs.append(a_bgra(img))
                            encontrada = True
                            break
                if encontrada:
                    break
            if encontrada:
                break
    return imgs


overlay_images = cargar_imagenes()
if not overlay_images:
    print(f"Aviso: no se encontraron imágenes 'mybad 1/2/3' en {os.path.join(BASE_DIR, 'img')}")

img_actual = None  # imagen ya redimensionada y rotada (se prepara una sola vez)
img_x = 0
img_y = 0


def rotar_imagen(foreground, angle):
    h_fg, w_fg = foreground.shape[:2]
    center = (w_fg // 2, h_fg // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h_fg * sin) + (w_fg * cos))
    new_h = int((h_fg * cos) + (w_fg * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(foreground, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def preparar_overlay(img, w, h, tam=200):
    ih, iw = img.shape[:2]
    escala = tam / max(ih, iw)
    redim = cv2.resize(img, (max(1, int(iw * escala)), max(1, int(ih * escala))))
    rotada = rotar_imagen(redim, random.uniform(-15, 15))
    fh, fw = rotada.shape[:2]
    margin_top = 120
    y = random.randint(margin_top, max(margin_top, h - fh - 50))
    x = random.randint(50, max(50, w - fw - 50))
    return rotada, x, y


def pegar_imagen(bg, fg, x, y):
    """Pega fg (BGRA) sobre bg en el sitio, con transparencia. Vectorizado (rápido)."""
    h_bg, w_bg = bg.shape[:2]
    fh, fw = fg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_bg, x + fw), min(h_bg, y + fh)
    if x1 >= x2 or y1 >= y2:
        return
    fg_roi = fg[y1 - y:y2 - y, x1 - x:x2 - x]
    roi = bg[y1:y2, x1:x2]
    alpha = fg_roi[:, :, 3:4].astype(np.float32) / 255.0
    mezcla = alpha * fg_roi[:, :, :3] + (1.0 - alpha) * roi
    roi[:] = mezcla.astype(np.uint8)


# ----------------------------------------------------------------------
# DETECCIÓN
# ----------------------------------------------------------------------
def es_mano_abierta(hand_landmarks, w, h):
    """Mano abierta = al menos 3 dedos con la punta más lejos de la muñeca que su articulación.
    Funciona aunque la mano esté inclinada o girada."""
    lm = hand_landmarks.landmark
    mx, my = lm[0].x * w, lm[0].y * h

    def dist(i):
        return math.hypot(lm[i].x * w - mx, lm[i].y * h - my)

    puntas = [8, 12, 16, 20]
    articulaciones = [6, 10, 14, 18]
    extendidos = sum(1 for p, a in zip(puntas, articulaciones) if dist(p) > dist(a))
    return extendidos >= 3


def puntos_dedos(hand_landmarks, w, h, ids=(4, 8, 12, 16, 20)):
    lm = hand_landmarks.landmark
    return np.array([[lm[i].x * w, lm[i].y * h] for i in ids], dtype=np.float32)


# --- Detección de dedos en 3D con histéresis ---
# Un dedo cuenta como levantado si cumple DOS condiciones:
#  1) Rectitud: distancia base->punta / largo total del dedo (recto ~1.0, garra ~0.75, puño ~0.45).
#  2) Lejanía: distancia de la punta al centro de la palma / tamaño de la palma.
#     Esto descarta los dedos a medio cerrar que quedan pegados a la palma (como en el gesto del cuadrado).
# Orden de los valores: pulgar, índice, medio, anular, meñique.
RECTITUD_ON = 0.68
RECTITUD_OFF = 0.55
DIST_ON = [0.50, 0.90, 0.85, 0.95, 0.80]   # para pasar a "levantado"
DIST_OFF = [0.40, 0.60, 0.55, 0.80, 0.65]  # por debajo de esto pasa a "cerrado"
FRAMES_ESTABLE = 2   # frames seguidos para que un dedo pase a LEVANTADO
FRAMES_APAGAR = 8    # frames seguidos para que un dedo pase a CERRADO (aguanta fallos al retorcer)
DEDOS_USADOS = (0, 1, 2)  # solo estos dedos forman la figura: pulgar, índice y medio

estado_dedos = [[False] * 5, [False] * 5, [False] * 5]  # slots 0 y 1: dos manos; slot 2: una sola mano
cont_dedos = [[0] * 5, [0] * 5, [0] * 5]
ultimos_valores = [None, None, None]  # (rectitudes, distancias) para mostrarlos en la ventana Trackers


def dedos_extendidos(hand_landmarks, w, h):
    """Devuelve (rectitudes, distancias): dos listas de 5 valores [pulgar, indice, medio, anular, menique]."""
    lm = hand_landmarks.landmark

    def P(i):
        return np.array([lm[i].x * w, lm[i].y * h, lm[i].z * w], dtype=np.float32)

    def rectitud(cadena):
        pts = [P(i) for i in cadena]
        total = sum(float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(len(pts) - 1))
        if total < 1e-6:
            return 0.0
        return float(np.linalg.norm(pts[-1] - pts[0])) / total

    centro = (P(0) + P(5) + P(9) + P(13) + P(17)) / 5.0
    palma = max(float(np.linalg.norm(P(0) - P(9))), 1e-6)

    rects = [rectitud((1, 2, 3, 4))]
    dists = [float(np.linalg.norm(P(4) - P(5))) / palma]  # pulgar: separación respecto al índice
    for cadena in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        rects.append(rectitud(cadena))
        dists.append(float(np.linalg.norm(P(cadena[-1]) - centro)) / palma)
    return rects, dists


def actualizar_dedos(slot, rects, dists):
    ultimos_valores[slot] = (rects, dists)
    for i in range(5):
        actual = estado_dedos[slot][i]
        if actual:
            deseado = rects[i] >= RECTITUD_OFF and dists[i] >= DIST_OFF[i]
        else:
            deseado = rects[i] > RECTITUD_ON and dists[i] > DIST_ON[i]
        if deseado == actual:
            cont_dedos[slot][i] = 0
        else:
            cont_dedos[slot][i] += 1
            limite = FRAMES_ESTABLE if deseado else FRAMES_APAGAR
            if cont_dedos[slot][i] >= limite:
                estado_dedos[slot][i] = deseado
                cont_dedos[slot][i] = 0


def reiniciar_dedos():
    for s_ in range(2):
        ultimos_valores[s_] = None
        for i in range(5):
            estado_dedos[s_][i] = False
            cont_dedos[s_][i] = 0


def triangulos_cara(cara):
    """Divide una cara de 4 puntos en 2 triángulos. Así no se rompe cuando el prisma se retuerce."""
    a, b, c, d = cara
    return [np.array([a, b, c], dtype=np.int32), np.array([a, c, d], dtype=np.int32)]


def dibujar_debug_mano(img, hl, slot, w, h):
    tx = max(5, int(hl.landmark[0].x * w) - 70)
    ty = min(h - 60, int(hl.landmark[0].y * h) + 25)
    letras = " ".join(c if estado_dedos[slot][i] else "-" for i, c in enumerate("PIMAQ"))
    cv2.putText(img, letras, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if ultimos_valores[slot] is not None:
        r_, d_ = ultimos_valores[slot]
        cv2.putText(img, "R " + " ".join(f"{int(v * 100):3d}" for v in r_),
                    (tx, ty + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(img, "D " + " ".join(f"{int(v * 100):3d}" for v in d_),
                    (tx, ty + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)


def suavizar(prev, nuevo, a=0.6):
    if prev is None:
        return nuevo
    return prev * (1 - a) + nuevo * a


def ajustar_a_salida(frame, W, H):
    """Mete el frame en el tamaño de la cámara virtual sin deformarlo (barras negras si hace falta)."""
    h, w = frame.shape[:2]
    if (w, h) == (W, H):
        return frame
    escala = min(W / w, H / h)
    nw, nh = max(1, int(w * escala)), max(1, int(h * escala))
    redim = cv2.resize(frame, (nw, nh))
    lienzo = np.zeros((H, W, 3), dtype=np.uint8)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = redim
    return lienzo


# ----------------------------------------------------------------------
# INTERFAZ (solo en la ventana de previsualización, NO sale por la cámara virtual)
# ----------------------------------------------------------------------
def dibujar_boton(img, rect, texto, color, dx=5):
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 2)
    cv2.putText(img, texto, (rect[0] + dx, rect[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def dibujar_botones(img):
    gris = (50, 50, 50)
    verde = (0, 255, 0)
    dibujar_boton(img, btn_left, "< Rotar", gris)
    dibujar_boton(img, btn_right, "Rotar >", gris)
    dibujar_boton(img, btn_mirror, "Espejo", verde if mirror_mode else gris, dx=10)
    dibujar_boton(img, btn_tracker, "Trackers", verde if show_debug_window else gris)


def dentro(rect, x, y):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def mouse_callback(event, x, y, flags, param):
    global rotation_state, mirror_mode, show_debug_window, show_buttons

    if event == cv2.EVENT_MOUSEMOVE:
        show_buttons = y <= 80

    if event == cv2.EVENT_LBUTTONDOWN:
        if dentro(btn_left, x, y):
            rotation_state = (rotation_state - 1) % 4
        elif dentro(btn_right, x, y):
            rotation_state = (rotation_state + 1) % 4
        elif dentro(btn_mirror, x, y):
            mirror_mode = not mirror_mode
        elif dentro(btn_tracker, x, y):
            show_debug_window = not show_debug_window
            if not show_debug_window:
                try:
                    cv2.destroyWindow(VENTANA_DEBUG)
                except cv2.error:
                    pass


cv2.namedWindow(VENTANA)
cv2.setMouseCallback(VENTANA, mouse_callback)

# --- Dibujo con un solo dedo (índice) / borrar con el puño (solo con UNA mano) ---
COLOR_TRAZO = (0, 200, 255)  # BGR (naranja)
GROSOR_TRAZO = 5
GRACIA_PEN = 0.35     # segundos que se aguanta un fallo de detección sin cortar el trazo
LIMITE_SALTO = 150    # píxeles máximos entre dos puntos para unirlos con línea
TIEMPO_BORRAR = 0.35  # segundos con el puño cerrado para borrar todo

lienzo = None
lienzo_mask = None
hay_trazos = False
estado_vista = None
dibujando = False
pen_ultimo = None
pen_suav = None
t_ultimo_gesto = 0.0
puno_desde = None
cursor_prev = None

# Puntos suavizados de cada mano (0 = izquierda de la imagen, 1 = derecha)
suav = [None, None]
ult_munecas = [None, None]  # última posición de cada muñeca (para no confundir las manos al cruzarlas)
ultimo_frame = None

# ----------------------------------------------------------------------
# BUCLE PRINCIPAL
# ----------------------------------------------------------------------
try:
    with pyvirtualcam.Camera(width=width, height=height, fps=fps, backend='unitycapture',
                             fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        print(f'Transmitiendo a Unity Capture en vivo: {cam.device}')

        while True:
            frame = camara.leer()

            # Si aún no hay frame nuevo, no reprocesamos (ahorra CPU y evita duplicados)
            if frame is None or frame is ultimo_frame:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                time.sleep(0.002)
                continue
            ultimo_frame = frame

            if mirror_mode:
                frame = cv2.flip(frame, 1)

            if rotation_state == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_state == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_state == 3:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            frame = np.ascontiguousarray(frame)
            h, w = frame.shape[:2]
            ahora = time.time()

            if lienzo is None or lienzo.shape[:2] != (h, w) or estado_vista != (mirror_mode, rotation_state):
                lienzo = np.zeros((h, w, 3), dtype=np.uint8)
                lienzo_mask = np.zeros((h, w), dtype=np.uint8)
                hay_trazos = False
                estado_vista = (mirror_mode, rotation_state)
                pen_ultimo = None
                pen_suav = None

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)

            debug_frame = frame.copy() if show_debug_window else None

            manos = []
            if results.multi_hand_landmarks:
                # Siempre en el mismo orden (izquierda -> derecha) para que el prisma no se cruce
                manos = sorted(results.multi_hand_landmarks, key=lambda m: m.landmark[0].x)
                if len(manos) == 2:
                    mu = [np.array([m.landmark[0].x * w, m.landmark[0].y * h]) for m in manos]
                    if ult_munecas[0] is not None:
                        directo = np.linalg.norm(mu[0] - ult_munecas[0]) + np.linalg.norm(mu[1] - ult_munecas[1])
                        cruzado = np.linalg.norm(mu[0] - ult_munecas[1]) + np.linalg.norm(mu[1] - ult_munecas[0])
                        if cruzado < directo:
                            manos.reverse()
                            mu.reverse()
                    ult_munecas[0], ult_munecas[1] = mu
                if debug_frame is not None:
                    for hl in manos:
                        mp_draw.draw_landmarks(debug_frame, hl, mp_hands.HAND_CONNECTIONS)

            # --- IMAGEN ALEATORIA CON UNA MANO ABIERTA ---
            una_mano_abierta = len(manos) == 1 and es_mano_abierta(manos[0], w, h)
            if una_mano_abierta:
                t_ultima_abierta = ahora
                if img_actual is None and overlay_images:
                    img_actual, img_x, img_y = preparar_overlay(random.choice(overlay_images), w, h)
            elif img_actual is not None and ahora - t_ultima_abierta > GRACIA_IMG:
                img_actual = None

            # --- DELAY DE 1 SEGUNDO PARA LAS 2 MANOS (con tolerancia a fallos breves) ---
            if len(manos) == 2:
                ultimo_visto_dos = ahora
                if two_hands_timer is None:
                    two_hands_timer = ahora
                elif ahora - two_hands_timer >= delay_required:
                    effect_active = True
            elif two_hands_timer is not None and ahora - ultimo_visto_dos > GRACIA_MANOS:
                two_hands_timer = None
                effect_active = False
                suav = [None, None]
                reiniciar_dedos()
                ult_munecas[0] = ult_munecas[1] = None

            # --- EFECTO PRISMA ---
            if effect_active and len(manos) == 2:
                for slot in (0, 1):
                    rects, dists = dedos_extendidos(manos[slot], w, h)
                    actualizar_dedos(slot, rects, dists)
                if debug_frame is not None:
                    for slot, hl in enumerate(manos):
                        tx = max(5, int(hl.landmark[0].x * w) - 70)
                        ty = min(h - 60, int(hl.landmark[0].y * h) + 25)
                        letras = " ".join(c if estado_dedos[slot][i] else "-" for i, c in enumerate("PIMAQ"))
                        cv2.putText(debug_frame, letras, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        if ultimos_valores[slot] is not None:
                            r_, d_ = ultimos_valores[slot]
                            cv2.putText(debug_frame, "R " + " ".join(f"{int(v * 100):3d}" for v in r_),
                                        (tx, ty + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                            cv2.putText(debug_frame, "D " + " ".join(f"{int(v * 100):3d}" for v in d_),
                                        (tx, ty + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                suav[0] = suavizar(suav[0], puntos_dedos(manos[0], w, h))
                suav[1] = suavizar(suav[1], puntos_dedos(manos[1], w, h))
                p1 = [tuple(p) for p in suav[0].astype(np.int32)]
                p2 = [tuple(p) for p in suav[1].astype(np.int32)]

                # Solo se usan pulgar, índice y medio, y solo los que están levantados en LAS DOS manos
                activos = [i for i in DEDOS_USADOS if estado_dedos[0][i] and estado_dedos[1][i]]
                n = len(activos)

                if n >= 1:
                    a1 = [p1[i] for i in activos]
                    a2 = [p2[i] for i in activos]

                    # Pares de dedos vecinos: 2 dedos -> cuadro, 3 dedos -> prisma triangular
                    pares = [(k, k + 1) for k in range(n - 1)]
                    if n == 3:
                        pares.append((2, 0))

                    if n >= 2:
                        caras = [[a1[i], a1[j], a2[j], a2[i]] for i, j in pares]

                        # Interior en gris = unión de las caras (y las dos tapas triangulares)
                        mask = np.zeros((h, w), dtype=np.uint8)
                        for cara in caras:
                            cv2.fillPoly(mask, triangulos_cara(cara), 255)
                        if n == 3:
                            cv2.fillPoly(mask, [np.array(a1, dtype=np.int32), np.array(a2, dtype=np.int32)], 255)
                        gris = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                        frame = np.where(mask[:, :, None] > 0, gris, frame).astype(np.uint8)

                        # Una cara de color por cada par de dedos (BGR)
                        colores_caras = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
                        overlay_color = frame.copy()
                        for k, cara in enumerate(caras):
                            cv2.fillPoly(overlay_color, triangulos_cara(cara), colores_caras[k % len(colores_caras)])
                        frame = cv2.addWeighted(overlay_color, 0.35, frame, 0.65, 0)

                        # Bordes de cada mano (el cuadro o el triángulo de cada lado)
                        for i, j in pares:
                            cv2.line(frame, a1[i], a1[j], (255, 255, 255), 2)
                            cv2.line(frame, a2[i], a2[j], (255, 255, 255), 2)

                    # Aristas que unen cada dedo de una mano con el de la otra
                    for k in range(n):
                        cv2.line(frame, a1[k], a2[k], (0, 255, 255), 2)

            # --- DIBUJAR CON EL ÍNDICE / BORRAR CON EL PUÑO (solo con una mano) ---
            cursor_prev = None
            if len(manos) == 1:
                hl = manos[0]
                rects, dists = dedos_extendidos(hl, w, h)
                actualizar_dedos(2, rects, dists)
                e = estado_dedos[2]
                lm = hl.landmark
                # Gesto del pincel: índice levantado (el pulgar da igual).
                # Para EMPEZAR: medio, anular y meñique cerrados.
                # Para SEGUIR: se tolera que uno de ellos parpadee como "levantado".
                otros = int(e[2]) + int(e[3]) + int(e[4])
                gesto_pen = e[1] and (otros <= 1 if dibujando else otros == 0)
                if gesto_pen:
                    dibujando = True
                    t_ultimo_gesto = ahora
                elif ahora - t_ultimo_gesto > GRACIA_PEN:
                    # Solo se corta el trazo si el gesto falla más de GRACIA_PEN segundos
                    dibujando = False
                    pen_ultimo = None
                    pen_suav = None
                puno = not (e[1] or e[2] or e[3] or e[4])

                if debug_frame is not None:
                    dibujar_debug_mano(debug_frame, hl, 2, w, h)

                if dibujando and gesto_pen:
                    punta = np.array([lm[8].x * w, lm[8].y * h], dtype=np.float32)
                    pen_suav = suavizar(pen_suav, punta, 0.2)
                    pt = (int(pen_suav[0]), int(pen_suav[1]))
                    if pen_ultimo is None:
                        cv2.circle(lienzo, pt, GROSOR_TRAZO // 2 + 1, COLOR_TRAZO, -1, cv2.LINE_AA)
                        cv2.circle(lienzo_mask, pt, GROSOR_TRAZO // 2 + 1, 255, -1, cv2.LINE_AA)
                        hay_trazos = True
                    elif math.hypot(pt[0] - pen_ultimo[0], pt[1] - pen_ultimo[1]) < LIMITE_SALTO:
                        cv2.line(lienzo, pen_ultimo, pt, COLOR_TRAZO, GROSOR_TRAZO, cv2.LINE_AA)
                        cv2.line(lienzo_mask, pen_ultimo, pt, 255, GROSOR_TRAZO, cv2.LINE_AA)
                        hay_trazos = True
                    pen_ultimo = pt
                    cursor_prev = pt

                # Puño cerrado sostenido -> borra todo
                if puno:
                    if puno_desde is None:
                        puno_desde = ahora
                    elif ahora - puno_desde >= TIEMPO_BORRAR and hay_trazos:
                        lienzo[:] = 0
                        lienzo_mask[:] = 0
                        hay_trazos = False
                else:
                    puno_desde = None
            else:
                if ahora - t_ultimo_gesto > GRACIA_PEN:
                    dibujando = False
                    pen_ultimo = None
                    pen_suav = None
                puno_desde = None

            if hay_trazos:
                alfa = lienzo_mask[:, :, None].astype(np.float32) / 255.0
                frame = (lienzo.astype(np.float32) * alfa + frame.astype(np.float32) * (1.0 - alfa)).astype(np.uint8)

            if img_actual is not None:
                pegar_imagen(frame, img_actual, img_x, img_y)

            # --- SALIDA A LA CÁMARA VIRTUAL (limpia, sin botones) ---
            cam.send(ajustar_a_salida(frame, width, height))

            # --- VENTANAS DE PREVISUALIZACIÓN (los botones solo se ven aquí) ---
            if show_buttons or cursor_prev is not None:
                preview = frame.copy()
                if show_buttons:
                    dibujar_botones(preview)
                if cursor_prev is not None:
                    cv2.circle(preview, cursor_prev, 9, (255, 255, 255), 2)
            else:
                preview = frame
            cv2.imshow(VENTANA, preview)

            if show_debug_window and debug_frame is not None:
                cv2.imshow(VENTANA_DEBUG, debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            # Cerrar con la X de la ventana
            if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                break
finally:
    camara.detener()
    hands.close()
    cv2.destroyAllWindows()
#  & "C:\Users\ns875\Documents\Nueva carpeta\venv\Scripts\python.exe" "C:\Users\ns875\Documents\Nueva carpeta\Traquer\Traquer.py"