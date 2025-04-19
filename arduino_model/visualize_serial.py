import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time

# Ustawienia
PORT = "/dev/ttyACM0"    # ← zamień na swój port np. "COM5" w Windows
BAUD = 115200
FS = 128
WINDOW_SEC = 10
MAX_POINTS = FS * WINDOW_SEC

# Bufory danych
ydata = deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS)
xdata = deque([i/FS for i in range(MAX_POINTS)], maxlen=MAX_POINTS)

# Serial
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# Wykres
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=1.5, color='red')

def init():
    ax.set_xlim(0, WINDOW_SEC)
    ax.set_ylim(-4.0, 4.0)  # typowe mV dla MLII
    return line,

def update(frame):
    try:
        line_str = ser.readline().decode().strip()
        value = float(line_str)

        # FILTR: odrzucaj outliery > 5 mV
        if abs(value) > 5:
            return line,

        ydata.append(value)
        xdata.append(xdata[-1] + 1/FS)
    except:
        return line,

    line.set_data(xdata, ydata)
    ax.set_xlim(xdata[0], xdata[-1])
    return line,

ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=1000/FS)
plt.title("ECG lead II (line, filtered, sliding 10s)")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda [mV]")
plt.tight_layout()
plt.show()

ser.close()
