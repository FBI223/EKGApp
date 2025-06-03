
import SwiftUI

struct InfoView: View {
    enum Tab: String, CaseIterable {
        case beat = "Beat"
        case rhythm = "Rhythm"
    }

    @State private var selectedTab: Tab = .beat

    private let beatInfos: [(code: String, description: String)] = [
        ("N", "Normal beat — normal QRS complex."),
        ("S", "Supraventricular beat — originating above ventricles."),
        ("V", "Ventricular beat — originating in ventricles."),
        ("F", "Fusion beat — fusion of normal and ventricular beat."),
        ("Q", "Unknown beat — unclassified or noise.")
    ]

    private let rhythmInfos: [(code: String, description: String)] = [
        ("NSR", "Normal Sinus Rhythm — normal heart rhythm."),
        ("AF_FLUTTER", "Atrial Fibrillation/Flutter — irregular and rapid atrial rhythm."),
        ("PAC", "Premature Atrial Contractions — early atrial beats."),
        ("PVC", "Premature Ventricular Contractions — early ventricular beats."),
        ("BBB", "Bundle Branch Block — delay/block in conduction pathway."),
        ("SVT", "Supraventricular Tachycardia — rapid atrial heartbeat."),
        ("AV_BLOCK", "Atrioventricular Block — impaired conduction atria to ventricles."),
        ("TORSADES", "Torsades de Pointes — dangerous ventricular tachycardia.")
    ]

    var body: some View {
        NavigationView {
            VStack {
                Picker("Select category", selection: $selectedTab) {
                    ForEach(Tab.allCases, id: \.self) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .padding()

                List {
                    if selectedTab == .beat {
                        ForEach(beatInfos, id: \.code) { info in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(info.code).font(.headline)
                                Text(info.description).font(.subheadline).foregroundColor(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    } else {
                        ForEach(rhythmInfos, id: \.code) { info in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(info.code).font(.headline)
                                Text(info.description).font(.subheadline).foregroundColor(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
                .listStyle(.plain)
            }
            .navigationTitle("ECG Info")
        }
    }
}

