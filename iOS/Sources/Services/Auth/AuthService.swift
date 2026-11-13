import Foundation
import Security
import UIKit

// Authentication state shared via @Environment.
// APIClient owns token storage; this class only mirrors "is there a token" for the UI.
@MainActor
@Observable
final class AuthService {
    private(set) var isLoggedIn: Bool = false

    private let api = APIClient.shared

    init() {
        #if DEBUG
        // UI tests launch with this flag so every run starts signed out.
        if ProcessInfo.processInfo.arguments.contains("-UITestResetAuth") {
            keychainDelete(key: "access_token")
            keychainDelete(key: "refresh_token")
        }
        #endif
        isLoggedIn = keychainLoad(key: "access_token") != nil
        #if DEBUG
        if DemoMode.isOn && DemoMode.initialTab != nil { isLoggedIn = true; return }
        #endif
        setupForegroundRefresh()
    }

    func login(email: String, password: String) async throws {
        _ = try await api.login(email: email, password: password)
        isLoggedIn = true
    }

    func register(email: String, password: String, nickname: String) async throws {
        _ = try await api.register(email: email, password: password, nickname: nickname)
        isLoggedIn = true
    }

    /// Re-reads the Keychain (APIClient deletes the tokens when a refresh fails).
    func refreshSessionState() {
        #if DEBUG
        if DemoMode.isOn { return }
        #endif
        isLoggedIn = keychainLoad(key: "access_token") != nil
    }

    func logout() {
        isLoggedIn = false
        Task { await api.logout() }
    }

    // MARK: - Private

    private func setupForegroundRefresh() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            // APIClient clears the Keychain when a refresh fails; reflect that here.
            MainActor.assumeIsolated { self?.isLoggedIn = keychainLoad(key: "access_token") != nil }
        }
    }
}

// MARK: - Keychain helpers (module-level functions, shared with APIClient)

private let keychainService = "com.shuttl.app"

func keychainSave(key: String, value: String) {
    let data = Data(value.utf8)
    let query: [CFString: Any] = [
        kSecClass: kSecClassGenericPassword,
        kSecAttrService: keychainService,
        kSecAttrAccount: key,
    ]
    SecItemDelete(query as CFDictionary)
    var add = query
    add[kSecValueData] = data
    SecItemAdd(add as CFDictionary, nil)
}

func keychainLoad(key: String) -> String? {
    let query: [CFString: Any] = [
        kSecClass: kSecClassGenericPassword,
        kSecAttrService: keychainService,
        kSecAttrAccount: key,
        kSecReturnData: true,
        kSecMatchLimit: kSecMatchLimitOne,
    ]
    var result: AnyObject?
    guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
          let data = result as? Data else { return nil }
    return String(data: data, encoding: .utf8)
}

func keychainDelete(key: String) {
    let query: [CFString: Any] = [
        kSecClass: kSecClassGenericPassword,
        kSecAttrService: keychainService,
        kSecAttrAccount: key,
    ]
    SecItemDelete(query as CFDictionary)
}
