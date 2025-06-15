import Foundation
import CoreBluetooth

class EKGBLEManager: NSObject, ObservableObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    @Published var rawBuffer = [Float]()
    private var tempBuffer = [Float]()
    @Published var isDeviceValid = false
    @Published var devices: [CBPeripheral] = []
    @Published var connectedPeripheral: CBPeripheral?
    @Published var statusMessage = "Waiting for Bluetooth..."
    @Published var eventLogs: [String] = []


    private var central: CBCentralManager!
    
    
    struct UUIDPair {
        let service: CBUUID
        let characteristic: CBUUID
    }

    private let allowedUUIDPairs: [UUIDPair] = [
        UUIDPair(
            service: CBUUID(string: "bd37e8b4-1bcf-4f42-bdd1-bebea1a51a1a"),
            characteristic: CBUUID(string: "7a1e8b7d-9a3e-4657-927b-339adddc2a5b")
        ),
        UUIDPair(
            service: CBUUID(string: "410de486-ba6e-47a9-85c8-715700eba0fa"),
            characteristic: CBUUID(string: "9e79707a-a0aa-4e79-9009-96d643ef755a")
        )
    ]

    
    private let allowedUUIDs: [(service: CBUUID, characteristic: CBUUID)] = [
        (
            service: CBUUID(string: "bd37e8b4-1bcf-4f42-bdd1-bebea1a51a1a"),
            characteristic: CBUUID(string: "7a1e8b7d-9a3e-4657-927b-339adddc2a5b")
        ),
        (
            service: CBUUID(string: "410de486-ba6e-47a9-85c8-715700eba0fa"),
            characteristic: CBUUID(string: "9e79707a-a0aa-4e79-9009-96d643ef755a")
        )
    ]

    
    
    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: nil)
    }
    
    
    func startScanWithTimeoutAfterConnect(timeout: TimeInterval = 1.5, onInvalid: @escaping () -> Void) {
        rawBuffer.removeAll()
        tempBuffer.removeAll()
        isDeviceValid = false

        DispatchQueue.main.asyncAfter(deadline: .now() + timeout) {
            if self.tempBuffer.isEmpty {
                self.disconnect()
                onInvalid()
            } else {
                self.rawBuffer = self.tempBuffer
                self.tempBuffer.removeAll()
                self.isDeviceValid = true
                self.log("✅ Device validated, data will now be processed")
            }
        }
    }



    func reset() {
        stopScan()
        disconnect()
        rawBuffer.removeAll()
        isDeviceValid = false
        devices.removeAll()
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
        let serviceUUIDs = allowedUUIDPairs.map { $0.service }
        p.discoverServices(serviceUUIDs)

    }

    func peripheral(_ p: CBPeripheral, didDiscoverServices error: Error?) {
        p.services?.forEach { service in
            if let pair = allowedUUIDPairs.first(where: { $0.service == service.uuid }) {
                p.discoverCharacteristics([pair.characteristic], for: service)
            }
        }
    }

    func peripheral(_ p: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        service.characteristics?.forEach { char in
            if let servicePair = allowedUUIDPairs.first(where: { $0.service == service.uuid && $0.characteristic == char.uuid }) {
                p.setNotifyValue(true, for: char)
                log("📡 Subscribed to EKG characteristic for service \(servicePair.service.uuidString)")
            }
        }
    }

    func peripheral(_ p: CBPeripheral, didUpdateValueFor char: CBCharacteristic, error: Error?) {
        guard let data = char.value else { return }
        
        var value: Float = 0
        _ = withUnsafeMutableBytes(of: &value) { data.copyBytes(to: $0) }
        DispatchQueue.main.async {
            if self.isDeviceValid {
                self.rawBuffer.append(value)
            } else {
                self.tempBuffer.append(value)
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

