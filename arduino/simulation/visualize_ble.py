import asyncio
from bleak import BleakClient, BleakScanner
import struct
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

SERVICE_UUID = "bd37e8b4-1bcf-4f42-bdd1-bebea1a51a1a"
CHARACTERISTIC_UUID = "7a1e8b7d-9a3e-4657-927b-339adddc2a5b"
TARGET_NAME = "ESP32_EKG"

FS = 128
WINDOW_SEC = 10
MAX_POINTS = FS * WINDOW_SEC

def plot_process(q: Queue):
    xdata = deque([i / FS for i in range(MAX_POINTS)], maxlen=MAX_POINTS)
    ydata = deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=1.5, color='red')
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlim(0, WINDOW_SEC)
    plt.title("BLE ECG lead II")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [mV]")

    def update(_):
        while not q.empty():
            val = q.get()
            ydata.append(val)
            xdata.append(xdata[-1] + 1 / FS)
        line.set_data(xdata, ydata)
        ax.set_xlim(xdata[0], xdata[-1])
        return line,

    ani = animation.FuncAnimation(fig, update, interval=1000 / FS, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

def handle_notify_factory(queue: Queue):
    def handler(_, data: bytearray):
        if len(data) == 4:
            mv = struct.unpack('<f', data)[0]
            if abs(mv) < 5:
                queue.put(mv)
    return handler

async def auto_connect_to_esp32():
    print(f"🔍 Szukanie urządzenia BLE o nazwie \"{TARGET_NAME}\"...")
    devices = await BleakScanner.discover(timeout=5.0)

    for d in devices:
        if d.name and d.name == TARGET_NAME:
            print(f"✅ Znaleziono: {d.name} — {d.address}")
            return d

    print("❌ Nie znaleziono urządzenia ESP32_EKG.")
    return None

async def ble_loop(queue: Queue):
    device = await auto_connect_to_esp32()
    if not device:
        return

    async with BleakClient(device.address) as client:
        if not client.is_connected:
            print("❌ Nie udało się połączyć.")
            return

        print("✅ Połączono! Subskrypcja danych EKG...")
        await client.start_notify(CHARACTERISTIC_UUID, handle_notify_factory(queue))

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("⛔ Zatrzymano BLE loop")
        finally:
            await client.stop_notify(CHARACTERISTIC_UUID)
            await client.disconnect()

def main():
    queue = Queue()

    plotter = Process(target=plot_process, args=(queue,))
    plotter.start()

    try:
        asyncio.run(ble_loop(queue))
    except KeyboardInterrupt:
        print("\n🛑 Przerwano przez użytkownika (Ctrl+C)")
    finally:
        plotter.terminate()
        plotter.join()

if __name__ == "__main__":
    main()
