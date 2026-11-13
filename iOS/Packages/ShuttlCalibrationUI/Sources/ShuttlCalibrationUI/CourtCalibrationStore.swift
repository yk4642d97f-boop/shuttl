import Foundation
import ShuttlCore

/// Stores the most recent manual corner taps independently for each upright
/// video resolution. Reusing a resolution avoids accidentally applying points
/// from a different crop or rotation.
public final class CourtCalibrationStore: @unchecked Sendable {
    public static let shared = CourtCalibrationStore()

    private let defaults: UserDefaults
    private let prefix: String
    private let lock = NSLock()

    public init(defaults: UserDefaults = .standard, keyPrefix: String = "shuttl.courtCalibration") {
        self.defaults = defaults
        self.prefix = keyPrefix
    }

    public func save(_ taps: CourtCornerTaps) throws {
        let data = try JSONEncoder().encode(taps)
        lock.lock()
        defer { lock.unlock() }
        defaults.set(data, forKey: key(for: taps.imageSize))
    }

    public func load(for imageSize: ImageSize) -> CourtCornerTaps? {
        lock.lock()
        defer { lock.unlock() }
        guard let data = defaults.data(forKey: key(for: imageSize)),
              let taps = try? JSONDecoder().decode(CourtCornerTaps.self, from: data),
              taps.imageSize == imageSize else { return nil }
        return taps
    }

    public func remove(for imageSize: ImageSize) {
        lock.lock()
        defer { lock.unlock() }
        defaults.removeObject(forKey: key(for: imageSize))
    }

    public func removeAll() {
        lock.lock()
        defer { lock.unlock() }
        for size in storedSizes() { defaults.removeObject(forKey: key(for: size)) }
    }

    private func key(for size: ImageSize) -> String {
        "\(prefix).\(size.width)x\(size.height)"
    }

    private func storedSizes() -> [ImageSize] {
        defaults.dictionaryRepresentation().keys.compactMap { key in
            guard key.hasPrefix(prefix + ".") else { return nil }
            let dimensions = key.dropFirst(prefix.count + 1).split(separator: "x")
            guard dimensions.count == 2, let width = Int(dimensions[0]), let height = Int(dimensions[1]) else { return nil }
            return ImageSize(width: width, height: height)
        }
    }
}
