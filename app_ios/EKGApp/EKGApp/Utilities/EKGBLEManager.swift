import Foundation
import CoreBluetooth

class EKGBLEManager: NSObject, ObservableObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    /// Incoming samples waiting to be consumed. Not @Published: appending at the sample
    /// rate would re-render every observing view. Consume it with `drainBuffer()`.
    private(set) var rawBuffer = [Float]()
    private var tempBuffer = [Float]()
    /// Upper bound on buffered samples (~30 s at 500 Hz) so the buffer cannot grow forever
    /// while nobody is consuming it (e.g. before Start is pressed).
    private let maxBufferedSamples = 15_000
    @Published var isDeviceValid = false
    @Published var devices: [CBPeripheral] = []
    @Published var connectedPeripheral: CBPeripheral?
    @Published var statusMessage = "Waiting for Bluetooth..."
    @Published var eventLogs: [String] = []
    /// True while a (real or simulated) device is selected; views switch to the signal UI on this.
    @Published var isConnected = false
    @Published var connectedDeviceName: String?

    /// When set, no Bluetooth is used: this signal is streamed in a loop at its own sample rate,
    /// exactly as if a validated BLE ECG device were sending it.
    let simulatedSignal: SimulatedSignal?
    var isSimulation: Bool { simulatedSignal != nil }
    private var simulationTimer: Timer?
    private var simulationStart: Date?
    private var simulationEmitted = 0
    private var simulationPosition = 0

    private var central: CBCentralManager?
    
    
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

    
    init(simulatedSignal: SimulatedSignal? = nil) {
        self.simulatedSignal = simulatedSignal
        super.init()
        // In simulation mode the stream starts on the view's onAppear → startScan()
        // (not here, to avoid publishing state changes during view construction).
        if simulatedSignal == nil {
            central = CBCentralManager(delegate: self, queue: nil)
        }
    }

    deinit {
        simulationTimer?.invalidate()
    }

    // MARK: - Simulation

    /// "Connects" to the simulated device and starts streaming samples.
    private func startSimulation() {
        guard let signal = simulatedSignal, simulationTimer == nil else { return }

        // Analyses derive window lengths and resampling from this setting
        if AppSettings.shared.sampleRateIn != signal.fs {
            AppSettings.shared.sampleRateIn = signal.fs
        }

        rawBuffer.removeAll()
        tempBuffer.removeAll()
        simulationStart = Date()
        simulationEmitted = 0
        isDeviceValid = true
        isConnected = true
        connectedDeviceName = "🧪 \(signal.name)"
        statusMessage = "Simulating \(signal.name) @ \(signal.fs) Hz"
        log("🧪 Simulation started: \(signal.name), \(signal.samples.count) samples @ \(signal.fs) Hz")

        // Emit in small packets (like batched BLE notifications); timing based on wall clock, so no drift
        let timer = Timer(timeInterval: 0.02, repeats: true) { [weak self] _ in
            self?.simulationTick()
        }
        RunLoop.main.add(timer, forMode: .common)
        simulationTimer = timer
    }

    private func stopSimulation() {
        guard simulationTimer != nil else { return }
        simulationTimer?.invalidate()
        simulationTimer = nil
        log("🧪 Simulation stopped")
    }

    private func simulationTick() {
        guard let signal = simulatedSignal, let start = simulationStart, !signal.samples.isEmpty else { return }
        let due = Int(Date().timeIntervalSince(start) * Double(signal.fs))
        var count = due - simulationEmitted
        guard count > 0 else { return }
        count = min(count, signal.fs * 2)   // after a long stall (e.g. app in background) skip ahead

        var packet = [Float]()
        packet.reserveCapacity(count)
        for _ in 0..<count {
            packet.append(signal.samples[simulationPosition])
            simulationPosition = (simulationPosition + 1) % signal.samples.count   // loop forever
        }
        simulationEmitted = due
        appendSamples(packet)
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



    /// Returns all buffered samples and empties the buffer.
    func drainBuffer() -> [Float] {
        let samples = rawBuffer
        rawBuffer.removeAll(keepingCapacity: true)
        return samples
    }

    /// Discards stale samples (call when starting a new stream).
    func clearBuffer() {
        rawBuffer.removeAll(keepingCapacity: true)
    }

    private func appendSamples(_ values: [Float]) {
        if isDeviceValid {
            rawBuffer.append(contentsOf: values)
            if rawBuffer.count > maxBufferedSamples {
                rawBuffer.removeFirst(rawBuffer.count - maxBufferedSamples)
            }
        } else {
            tempBuffer.append(contentsOf: values)
            if tempBuffer.count > maxBufferedSamples {
                tempBuffer.removeFirst(tempBuffer.count - maxBufferedSamples)
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
        log("🔁 BLE Manager reset")
    }

    func startScan() {
        if isSimulation {
            startSimulation()
            return
        }
        guard let central, central.state == .poweredOn else {
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
        guard let central else { return }
        central.stopScan()
        log("🛑 Stopped scanning")
    }

    func connect(to device: CBPeripheral) {
        guard let central else { return }
        stopScan()
        connectedPeripheral = device
        connectedDeviceName = device.name ?? "Unknown"
        isConnected = true
        device.delegate = self
        central.connect(device, options: nil)
        statusMessage = "Connecting to \(device.name ?? "Unknown")..."
        log("🔗 Connecting to \(device.name ?? "Unknown")")
    }

    func disconnect() {
        if isSimulation {
            stopSimulation()
            rawBuffer.removeAll()
            isDeviceValid = false
            isConnected = false
            connectedDeviceName = nil
            statusMessage = "Disconnected"
            return
        }

        if let peripheral = connectedPeripheral {
            peripheral.delegate = nil
            central?.cancelPeripheralConnection(peripheral)
        }
        connectedPeripheral = nil
        isConnected = false
        connectedDeviceName = nil
        rawBuffer.removeAll()
        devices.removeAll()
        statusMessage = "Disconnected"
        log("⛔ Disconnected")

        // Optional restart after short delay to ensure clean state
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.startScan()
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

    func centralManager(_ c: CBCentralManager, didDisconnectPeripheral p: CBPeripheral, error: Error?) {
        guard p.identifier == connectedPeripheral?.identifier else { return }
        log("⚠️ Device disconnected: \(p.name ?? "Unknown")")
        connectedPeripheral = nil
        isConnected = false
        connectedDeviceName = nil
        isDeviceValid = false
        statusMessage = "Device disconnected"
    }

    func centralManager(_ c: CBCentralManager, didFailToConnect p: CBPeripheral, error: Error?) {
        log("❌ Failed to connect: \(error?.localizedDescription ?? "unknown error")")
        connectedPeripheral = nil
        isConnected = false
        connectedDeviceName = nil
        statusMessage = "Connection failed"
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
        guard let data = char.value, data.count >= 4 else { return }

        // Packet = one or more little-endian Float32 samples (supports batched notifications)
        let count = data.count / 4
        var values = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for i in 0..<count {
                values[i] = raw.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
            }
        }
        let finite = values.filter { $0.isFinite }
        guard !finite.isEmpty else { return }

        DispatchQueue.main.async {
            self.appendSamples(finite)
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

