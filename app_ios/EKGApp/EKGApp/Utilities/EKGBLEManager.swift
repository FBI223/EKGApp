import Foundation
import CoreBluetooth

class EKGBLEManager: NSObject, ObservableObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    @Published var rawBuffer = [Float]()
    @Published var devices: [CBPeripheral] = []
    @Published var connectedPeripheral: CBPeripheral?
    @Published var statusMessage = "Waiting for Bluetooth..."
    @Published var eventLogs: [String] = []
    @Published var latestMetaMessage: String? = nil

    private var central: CBCentralManager!
    private let serviceUUID = CBUUID(string: "bd37e8b4-1bcf-4f42-bdd1-bebea1a51a1a")
    private let charUUID = CBUUID(string: "7a1e8b7d-9a3e-4657-927b-339adddc2a5b")

    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: nil)
    }

    func reset() {
        stopScan()
        disconnect()
        rawBuffer.removeAll()
        devices.removeAll()
        latestMetaMessage = nil
        connectedPeripheral = nil
        central.delegate = self
        log("🔁 BLE Manager reset")
    }

    func startScan() {
        guard central.state == .poweredOn else {
            statusMessage = "Bluetooth not available"
            log("❌ Cannot scan, Bluetooth not powered on")
            return
        }

        devices.removeAll()
        statusMessage = "Scanning..."
        log("🔍 Scanning for devices...")
        central.scanForPeripherals(withServices: nil, options: nil)
    }

    func stopScan() {
        central.stopScan()
        log("🛑 Stopped scanning")
    }

    func connect(to device: CBPeripheral) {
        stopScan()
        connectedPeripheral = device
        device.delegate = self
        central.connect(device, options: nil)
        statusMessage = "Connecting to \(device.name ?? "Unknown")..."
        log("🔗 Connecting to \(device.name ?? "Unknown")")
    }

    func disconnect() {
        if let peripheral = connectedPeripheral {
            peripheral.delegate = nil
            central.cancelPeripheralConnection(peripheral)
        }
        connectedPeripheral = nil
        rawBuffer.removeAll()
        devices.removeAll()
        latestMetaMessage = nil
        statusMessage = "Disconnected"
        log("⛔ Disconnected")

        // Optional restart after short delay to ensure clean state
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.startScan()
        }
    }

    func centralManagerDidUpdateState(_ c: CBCentralManager) {
        log("📶 Central state: \(c.state.rawValue)")
        if c.state == .poweredOn {
            startScan()
        } else {
            statusMessage = "Bluetooth not available"
        }
    }

    func centralManager(_ c: CBCentralManager, didDiscover p: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        if !devices.contains(where: { $0.identifier == p.identifier }) {
            devices.append(p)
            log("🔍 Found device: \(p.name ?? "Unknown") RSSI=\(RSSI)")
        }
    }

    func centralManager(_ c: CBCentralManager, didConnect p: CBPeripheral) {
        statusMessage = "Connected to \(p.name ?? "Unknown")"
        log("✅ Connected to \(p.name ?? "Unknown")")
        p.discoverServices([serviceUUID])
    }

    func peripheral(_ p: CBPeripheral, didDiscoverServices error: Error?) {
        p.services?.forEach { service in
            if service.uuid == serviceUUID {
                p.discoverCharacteristics([charUUID], for: service)
            }
        }
    }

    func peripheral(_ p: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        service.characteristics?.forEach { char in
            if char.uuid == charUUID {
                p.setNotifyValue(true, for: char)
                log("📡 Subscribed to EKG characteristic")
            }
        }
    }

    func peripheral(_ p: CBPeripheral, didUpdateValueFor char: CBCharacteristic, error: Error?) {
        guard let data = char.value else { return }

        if let text = String(data: data, encoding: .utf8), text.starts(with: "META;") {
            DispatchQueue.main.async {
                self.latestMetaMessage = text
            }
            log("📥 Received META")
        } else {
            var value: Float = 0
            _ = withUnsafeMutableBytes(of: &value) { data.copyBytes(to: $0) }
            DispatchQueue.main.async {
                self.rawBuffer.append(value)
            }
        }
    }

    func log(_ text: String) {
        DispatchQueue.main.async {
            self.eventLogs.append("[\(self.formattedTime())] \(text)")
            if self.eventLogs.count > 100 { self.eventLogs.removeFirst() }
        }
    }

    func formattedTime() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: Date())
    }
}

