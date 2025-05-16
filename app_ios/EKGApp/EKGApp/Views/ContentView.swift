import SwiftUI
import Charts
import Combine

struct ContentView: View {
    @State private var selectedMode: AnalysisMode? = nil

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("ECG Analysis Menu")
                    .font(.headline)

                NavigationLink(destination: BeatAnalysisView()) {
                    Text("🔎 Beat Analysis")
                        .font(.title2)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }

                NavigationLink(destination: RhythmAnalysisView()) {
                    Text("📈 Rhythm Analysis")
                        .font(.title2)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.purple)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }

                NavigationLink(destination: SettingsView()) {
                    Text("⚙️ Settings")
                        .font(.title3)
                        .padding()
                }
            }
            .padding()
            .navigationTitle("ECG App")
        }
    }
}

