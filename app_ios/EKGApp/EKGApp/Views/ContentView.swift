import SwiftUI
import Charts
import Combine

struct ContentView: View {
    public enum AnalysisMode {
        case beat
        case rhythm
    }

    @ObservedObject private var settings = AppSettings.shared
    var backgroundColor: Color { settings.darkModeEnabled ? .black : .white }
    var foregroundColor: Color { settings.darkModeEnabled ? .white : .black }

    var body: some View {
        NavigationView {
            VStack(spacing: 32) {
                // Górna ikona i tytuł
                VStack(spacing: 8) {
                    Image(systemName: "waveform.path.ecg.rectangle")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 60, height: 60)
                        .foregroundColor(.red)
                    Text("ECG App")
                        .font(.largeTitle)
                        .fontWeight(.semibold)
                        .foregroundColor(foregroundColor)
                }
                .padding(.top, 32)


                // Przyciski
                VStack(spacing: 16) {
                    NavigationLink(destination: BeatAnalysisView()) {
                        Text("📈 Beat Analysis")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0)) // HEX #630000
                            .foregroundColor(.white)
                            .cornerRadius(16)
                    }

                    NavigationLink(destination: RhythmAnalysisView()) {
                        Text("📈 Rhythm Analysis")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0)) // HEX #630000
                            .foregroundColor(.white)
                            .cornerRadius(16)
                    }

                    NavigationLink(destination: SettingsView()) {
                        Text("⚙️ Settings")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color(red: 0.39, green: 0.0, blue: 0.0)) // HEX #630000
                            .foregroundColor(foregroundColor)
                            .cornerRadius(16)
                    }
                }

                Spacer()
            }
            .padding()
            .background(backgroundColor.ignoresSafeArea())
        }
        .preferredColorScheme(settings.darkModeEnabled ? .dark : .light)
    }
}

