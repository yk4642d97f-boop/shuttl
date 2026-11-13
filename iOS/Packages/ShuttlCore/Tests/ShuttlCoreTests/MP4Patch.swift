import Foundation

enum MP4PatchError: Error { case malformed(String) }

/// Rewrites the video track's `stts` (decode-time table) so that samples `k` and `k+1`
/// share one timestamp, for every `k` in `at`. Apple's writers refuse duplicate PTS,
/// so this is the only way to get such a file in a test. The source must have no
/// `ctts` (no frame reordering), so PTS == DTS and the edit is a duplicate PTS.
func makeDuplicatePTS(_ src: URL, _ dst: URL, at dups: [Int]) throws {
    var d = [UInt8](try Data(contentsOf: src))
    func u32(_ o: Int) -> Int { Int(d[o]) << 24 | Int(d[o + 1]) << 16 | Int(d[o + 2]) << 8 | Int(d[o + 3]) }
    func put32(_ o: Int, _ v: Int) {
        d[o] = UInt8(v >> 24 & 0xff); d[o + 1] = UInt8(v >> 16 & 0xff); d[o + 2] = UInt8(v >> 8 & 0xff); d[o + 3] = UInt8(v & 0xff)
    }
    func type(_ o: Int) -> String { String(decoding: d[(o + 4)..<(o + 8)], as: UTF8.self) }
    func children(_ start: Int, _ end: Int) -> [(o: Int, s: Int)] {
        var out: [(o: Int, s: Int)] = []; var o = start
        while o + 8 <= end { let s = u32(o); guard s >= 8, o + s <= end else { break }; out.append((o, s)); o += s }
        return out
    }
    func child(_ b: (o: Int, s: Int), _ t: String) -> (o: Int, s: Int)? { children(b.o + 8, b.o + b.s).first { type($0.o) == t } }

    let top = children(0, d.count)
    guard let moov = top.first(where: { type($0.o) == "moov" }), let mdat = top.first(where: { type($0.o) == "mdat" })
    else { throw MP4PatchError.malformed("no moov/mdat") }
    let trak = try children(moov.o + 8, moov.o + moov.s).first { t in
        guard type(t.o) == "trak", let mdia = child(t, "mdia"), let hdlr = child(mdia, "hdlr") else { return false }
        return String(decoding: d[(hdlr.o + 16)..<(hdlr.o + 20)], as: UTF8.self) == "vide"
    } ?? { throw MP4PatchError.malformed("no video trak") }()
    guard let mdia = child(trak, "mdia"), let minf = child(mdia, "minf"), let stbl = child(minf, "stbl"),
          let stts = child(stbl, "stts") else { throw MP4PatchError.malformed("no stts") }
    if child(stbl, "ctts") != nil { throw MP4PatchError.malformed("source has ctts (frame reordering)") }

    // Expand to per-sample deltas, collapse each pair, re-encode as runs.
    var deltas: [Int] = []
    for e in 0..<u32(stts.o + 12) {
        let base = stts.o + 16 + 8 * e
        deltas += Array(repeating: u32(base + 4), count: u32(base))
    }
    for k in dups where k + 1 < deltas.count { deltas[k + 1] += deltas[k]; deltas[k] = 0 }
    var runs: [(count: Int, delta: Int)] = []
    for v in deltas { if runs.last?.delta == v { runs[runs.count - 1].count += 1 } else { runs.append((1, v)) } }
    var body: [UInt8] = Array(d[(stts.o + 8)..<(stts.o + 12)])            // version + flags
    func be(_ v: Int) -> [UInt8] { [UInt8(v >> 24 & 0xff), UInt8(v >> 16 & 0xff), UInt8(v >> 8 & 0xff), UInt8(v & 0xff)] }
    body += be(runs.count)
    for r in runs { body += be(r.count) + be(r.delta) }
    let newBox = be(8 + body.count) + Array("stts".utf8) + body
    let diff = newBox.count - stts.s

    // Chunk offsets point into mdat; they move only if mdat comes after moov.
    if mdat.o > moov.o {
        for t in children(moov.o + 8, moov.o + moov.s) where type(t.o) == "trak" {
            guard let m = child(t, "mdia"), let mi = child(m, "minf"), let sb = child(mi, "stbl") else { continue }
            if let stco = child(sb, "stco") {
                for i in 0..<u32(stco.o + 12) { put32(stco.o + 16 + 4 * i, u32(stco.o + 16 + 4 * i) + diff) }
            }
            if child(sb, "co64") != nil { throw MP4PatchError.malformed("co64 not supported") }
        }
    }
    for anc in [moov, trak, mdia, minf, stbl] { put32(anc.o, u32(anc.o) + diff) }   // all start before stts
    d.replaceSubrange(stts.o..<(stts.o + stts.s), with: newBox)
    try Data(d).write(to: dst)
}
