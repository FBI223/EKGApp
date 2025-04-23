import SwiftUI
import CoreBluetooth
import Accelerate
import CoreML
import UniformTypeIdentifiers

// MARK: - BLE EKG Manager

class EKGBLEManager: NSObject, ObservableObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    private var central: CBCentralManager!
    private var ekgPeripheral: CBPeripheral?
    private let serviceUUID = CBUUID(string: "bd37e8b4-1bcf-4f42-bdd1-bebea1a51a1a")
    private let charUUID    = CBUUID(string: "7a1e8b7d-9a3e-4657-927b-339adddc2a5b")
    @Published var rawBuffer = [Float]()
    
    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: nil)
    }
    
    func centralManagerDidUpdateState(_ c: CBCentralManager) {
        if c.state == .poweredOn {
            central.scanForPeripherals(withServices: [serviceUUID], options: nil)
        }
    }
    
    func centralManager(_ c: CBCentralManager, didDiscover p: CBPeripheral, advertisementData: [String:Any], rssi RSSI: NSNumber) {
        ekgPeripheral = p
        p.delegate = self
        central.stopScan()
        central.connect(p, options: nil)
    }
    
    func centralManager(_ c: CBCentralManager, didConnect p: CBPeripheral) {
        p.discoverServices([serviceUUID])
    }
    
    func peripheral(_ p: CBPeripheral, didDiscoverServices error: Error?) {
        p.services?.forEach { s in
            if s.uuid == serviceUUID {
                p.discoverCharacteristics([charUUID], for: s)
            }
        }
    }
    
    func peripheral(_ p: CBPeripheral, didDiscoverCharacteristicsFor s: CBService, error: Error?) {
        s.characteristics?.forEach { c in
            if c.uuid == charUUID {
                p.setNotifyValue(true, for: c)
            }
        }
    }
    
    func peripheral(_ p: CBPeripheral, didUpdateValueFor c: CBCharacteristic, error: Error?) {
        guard let data = c.value else { return }
        var v: Float = 0
        _ = withUnsafeMutableBytes(of: &v) { data.copyBytes(to: $0) }
        rawBuffer.append(v)
    }
}




