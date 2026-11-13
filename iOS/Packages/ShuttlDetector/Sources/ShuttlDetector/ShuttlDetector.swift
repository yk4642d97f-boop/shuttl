// ShuttlDetector — on-device port of backend/app/services/tracking/detector.py.
//
//   VideoSource ─► FrameGeometry.warp (cv2-exact letterbox, RGB canvas) ─► [BackgroundPlan median]
//   ─► windows of seq_len frames every `stride` ─► CoreMLHeatmapNetwork (TrackNetV3 / WASB export)
//   ─► overlapping-window fusion (ensembleWeights) ─► HeatmapDecoder (peak/2 blob centroid, top-k)
//   ─► FrameGeometry.peakToFull ─► [[ShuttleCandidate]] per frame, full-res upright px.
//
// Entry point: `HeatmapDetector.load(modelURL:)` / `.build(kind:modelsDirectory:)`, then
// `detectVideo(_:progress:)`.
