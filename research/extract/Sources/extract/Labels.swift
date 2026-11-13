import Foundation

/// One labelled rally of a ShuttleSet set CSV and the frame span extracted for it.
struct RallySpec: Codable {
    var set: Int
    var rally: Int
    var hitFrames: [Int]
    var start = 0
    var stop = 0

    var key: String { String(format: "s%d_r%03d", set, rally) }
}

enum Labels {
    /// Rallies of every `set*.csv` in `dir`, span = [first hit − padBefore, last hit + padAfter] ∩ the video.
    /// The tail is longer than the head because the shuttle keeps flying after the last hit and
    /// only reaches the floor (the label's landing point) a few seconds later.
    static func rallies(in dir: URL, fps: Double, nFrames: Int,
                        padBefore: Double = 2, padAfter: Double = 6) throws -> [RallySpec] {
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("set") && $0.pathExtension == "csv" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let before = Int((padBefore * fps).rounded()), after = Int((padAfter * fps).rounded())
        var out: [RallySpec] = []
        for f in files {
            let set = Int(f.deletingPathExtension().lastPathComponent.dropFirst(3)) ?? 0
            var order: [Int] = [], hits: [Int: [Int]] = [:]
            for row in try csvRows(f) {
                guard let rv = Double(row["rally"] ?? ""), let fr = Double(row["frame_num"] ?? "") else { continue }
                let r = Int(rv)
                if hits[r] == nil { order.append(r) }
                hits[r, default: []].append(Int(fr.rounded()))
            }
            for r in order {
                let h = hits[r]!.sorted()
                var s = RallySpec(set: set, rally: r, hitFrames: h)
                s.start = max(0, h.first! - before)
                s.stop = min(nFrames, h.last! + after + 1)
                if s.stop > s.start { out.append(s) }
            }
        }
        return out
    }

    static func csvRows(_ url: URL) throws -> [[String: String]] {
        let text = try String(contentsOf: url, encoding: .utf8)
        var lines = text.split(whereSeparator: \.isNewline).map { parse(String($0)) }
        guard !lines.isEmpty else { return [] }
        let header = lines.removeFirst().map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "\u{FEFF} ")) }
        return lines.map { Dictionary(zip(header, $0), uniquingKeysWith: { a, _ in a }) }
    }

    private static func parse(_ line: String) -> [String] {
        var fields: [String] = [], cur = "", quoted = false
        for ch in line {
            if ch == "\"" { quoted.toggle() } else if ch == ",", !quoted { fields.append(cur); cur = "" } else { cur.append(ch) }
        }
        fields.append(cur)
        return fields
    }
}
