import Foundation

/// Connection policy only; all views and presentation remain shared by both paths.
enum CloudConfiguration {
    static var baseURL: String {
        var configured = Bundle.main.object(forInfoDictionaryKey: "ShuttlAPIBaseURL") as? String
        #if DEBUG
        if let override = ProcessInfo.processInfo.environment["SHUTTL_API_BASE_URL"], !override.isEmpty {
            configured = override
        }
        #endif
        return (configured?.isEmpty == false ? configured! : "http://127.0.0.1:8000")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    static var isEnabled: Bool {
        acceptsCloud(enabled: Bundle.main.object(forInfoDictionaryKey: "ShuttlCloudAnalysisEnabled") as? Bool ?? false,
                     endpoint: baseURL)
    }

    static func acceptsCloud(enabled: Bool, endpoint: String) -> Bool {
        guard enabled, let url = URL(string: endpoint), url.scheme == "https",
              let host = url.host, !host.isEmpty, url.user == nil, url.password == nil else { return false }
        return true
    }
}
