import Foundation
import ShuttlPlayers

/// Recall / cost table of the layout study (stdout + JSON).
@MainActor
func writeLayoutReport(cells: [String: [String: Cell]], farPx: [String: [Double]], kOf: [String: [String: [Double]]],
                       setups: [String], timedNames: [String], out: URL) {
    var json: [String: Any] = [:]
    let passes: [String: Int] = ["none": 1, "single": 2, "lateral": 3, "lateral_batched": 3, "depth": 3]
    print("\n== far-crop layouts: Vision recall vs YOLO tracks, cost per sampled frame (release, this Mac) ==")
    for s in setups {
        guard let row = cells[s] else { continue }
        let px = farPx[s] ?? []
        print(String(format: "\n%@ — far player bbox median %.0f px (p10 %.0f, p90 %.0f)%@", s as NSString,
                     median(px), pct(px, 10), pct(px, 90), s == "broadcast" ? "" : " — near-side numbers not phone-realistic"))
        print("  layout           k   far det  far trk  far err m (p95)  sw | near det  near trk | ms/frame median (mean)  passes")
        var js: [String: Any] = ["far_player_px_median": median(px), "far_player_px_p10": pct(px, 10),
                                 "far_player_px_p90": pct(px, 90)]
        for name in timedNames {
            let layout = name == "lateral_batched" ? "lateral" : name
            guard let c = row[layout] else { continue }
            let ms = row[name]?.ms ?? []
            let f = c.far.summary, nr = c.near.summary
            let k = median(kOf[s]?[layout] ?? [])
            print(String(format: "  %-15@ %4.1f  %6.3f   %6.3f   %5.3f (%5.3f)  %3.0f |  %6.3f    %6.3f  |   %6.1f (%6.1f)        %d",
                         name as NSString, k, f["det_recall"]!, f["track_recall"]!, f["foot_err_median_m"]!,
                         f["foot_err_p95_m"]!, f["id_switches"]!, nr["det_recall"]!, nr["track_recall"]!,
                         median(ms), ms.isEmpty ? .nan : ms.reduce(0, +) / Double(ms.count), passes[name] ?? 0))
            js[name] = ["k": k, "far": f, "near": nr, "ms_median": median(ms),
                        "ms_mean": ms.isEmpty ? Double.nan : ms.reduce(0, +) / Double(ms.count),
                        "timed_frames": Double(ms.count), "vision_passes": Double(passes[name] ?? 0)]
        }
        json[s] = js
    }
    if let data = try? JSONSerialization.data(withJSONObject: clean(json), options: [.prettyPrinted, .sortedKeys]) {
        try? data.write(to: out)
        print("\nwrote", out.path)
    }
}
