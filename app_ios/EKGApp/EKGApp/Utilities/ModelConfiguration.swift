import CoreML

extension MLModelConfiguration {
    /// Configuration shared by all bundled models.
    /// The iOS Simulator has no MPSGraph (GPU) backend for CoreML: it falls back to CPU anyway and logs
    /// "E5RT ... MpsGraph backend validation on incompatible OS" on every prediction, so use CPU there.
    /// On a real device all compute units (CPU, GPU, Neural Engine) are used.
    static var appDefault: MLModelConfiguration {
        let config = MLModelConfiguration()
        #if targetEnvironment(simulator)
        config.computeUnits = .cpuOnly
        #else
        config.computeUnits = .all
        #endif
        return config
    }
}
