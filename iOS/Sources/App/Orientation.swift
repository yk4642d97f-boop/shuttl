import SwiftUI
import UIKit

// Recording and replay are shot and watched with the phone on its side, so those two
// screens may rotate; everything else is a portrait broadsheet and stays upright.

@MainActor
enum OrientationPolicy {
    private(set) static var mask: UIInterfaceOrientationMask = .portrait

    static func allowLandscape(_ allowed: Bool) {
        let newMask: UIInterfaceOrientationMask = allowed ? [.portrait, .landscape] : .portrait
        guard newMask != mask else { return }
        mask = newMask
        guard let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).first else { return }
        scene.keyWindow?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        if !allowed, scene.interfaceOrientation.isLandscape {
            scene.requestGeometryUpdate(.iOS(interfaceOrientations: .portrait)) { _ in }
        }
    }

    /// Rotates to landscape now (demo screenshots; the simulator can't be tilted from a script).
    static func requestLandscape() {
        guard let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).first else { return }
        scene.requestGeometryUpdate(.iOS(interfaceOrientations: .landscapeRight)) { _ in }
    }
}

final class ShuttlAppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        guard identifier == VideoUploadService.backgroundIdentifier else { completionHandler(); return }
        Task {
            await VideoUploadService.shared.handleBackgroundEvents { completionHandler() }
        }
    }

    func application(_ application: UIApplication,
                     supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        MainActor.assumeIsolated { OrientationPolicy.mask }
    }
}

extension View {
    /// True when an iPhone is on its side (compact height).
    func isLandscapeLayout(_ verticalSizeClass: UserInterfaceSizeClass?) -> Bool {
        verticalSizeClass == .compact
    }
}
