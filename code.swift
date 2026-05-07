import SwiftUI
import ARKit
import Combine
import UIKit

final class ARIntrinsicsDelegate: NSObject, ObservableObject, ARSessionDelegate {
    @Published var intrinsicsText: String = "Starting AR session..."
    @Published var statusText: String = "Move the phone a little if needed."

    let session = ARSession()
    private var hasCaptured = false

    override init() {
        super.init()
        session.delegate = self
        startSession()
    }

    func startSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            intrinsicsText = "ARKit world tracking is not supported on this device."
            statusText = "Use a newer iPhone/iPad."
            return
        }

        let config = ARWorldTrackingConfiguration()
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        hasCaptured = false
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let K = frame.camera.intrinsics

        let fx = K[0, 0]
        let fy = K[1, 1]
        let cx = K[2, 0]
        let cy = K[2, 1]

        let matrixString = """
        {
          "intrinsics": [
            [\(K[0,0]), \(K[1,0]), \(K[2,0])],
            [\(K[0,1]), \(K[1,1]), \(K[2,1])],
            [\(K[0,2]), \(K[1,2]), \(K[2,2])]
          ],
          "fx": \(fx),
          "fy": \(fy),
          "cx": \(cx),
          "cy": \(cy)
        }
        """

        DispatchQueue.main.async {
            self.intrinsicsText = matrixString
            self.statusText = "Done. Tap Copy Intrinsics and send it."
        }

        if !hasCaptured {
            hasCaptured = true
            session.pause()
        }
    }
}

struct ContentView: View {
    @StateObject private var tracker = ARIntrinsicsDelegate()

    var body: some View {
        VStack(spacing: 16) {
            Text("iPhone Camera Intrinsics")
                .font(.title2)
                .bold()

            Text(tracker.statusText)
                .font(.subheadline)
                .multilineTextAlignment(.center)

            ScrollView {
                Text(tracker.intrinsicsText)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .background(Color(.secondarySystemBackground))
            .cornerRadius(12)

            Button("Copy Intrinsics") {
                UIPasteboard.general.string = tracker.intrinsicsText
            }
            .buttonStyle(.borderedProminent)

            Button("Restart Capture") {
                tracker.startSession()
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }
}