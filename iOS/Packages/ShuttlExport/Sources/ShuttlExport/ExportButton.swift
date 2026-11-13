#if os(iOS)
import Photos
import SwiftUI
import ShuttlCore
import ShuttlDesign

/// One-tap "export a share video" control: idle → progress → share / save, with an
/// inline retry on failure. Looks like the rest of the app (ShuttlDesign components).
@available(iOS 17.0, *)
@MainActor
public struct ExportButton: View {
    private let video: URL
    private let analysis: MatchAnalysis
    private let options: ExportOptions

    @State private var outputURL: URL?
    @State private var phase: ExportPhase = .idle
    @State private var saveStatus: ExportSaveStatus = .none

    public init(video: URL, analysis: MatchAnalysis, options: ExportOptions = ExportOptions()) {
        self.video = video
        self.analysis = analysis
        self.options = options
    }

    public var body: some View {
        ExportPanel(phase: phase, saveStatus: saveStatus, start: beginExport, save: saveToPhotos) {
            if let outputURL {
                ShareLink(item: outputURL) { ExportShareLabel() }
                    .buttonStyle(PrintButtonStyle(shape: Capsule(), plate: .ink, depth: 4))
                    .accessibilityLabel("分享视频")
            }
        }
        .animation(AppAnimation.spring, value: phase)
        .animation(AppAnimation.standard, value: saveStatus)
        .onAppear { DisplayFace.register() }
    }

    private func beginExport() {
        if case .exporting = phase { return }
        phase = .exporting(0)
        saveStatus = .none
        outputURL = nil
        Task {
            do {
                let result = try await OverlayExporter.export(video: video, analysis: analysis, options: options) { value in
                    Task { @MainActor in
                        if case .exporting = phase { phase = .exporting(value) }
                    }
                }
                outputURL = result
                phase = .done
            } catch is CancellationError {
                phase = .failed("导出已取消")
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    private func saveToPhotos() {
        guard let outputURL, saveStatus != .saving else { return }
        saveStatus = .saving
        Task {
            let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            guard status == .authorized || status == .limited else {
                saveStatus = .failed("没有照片图库添加权限，请在系统设置中允许 Shuttl 保存视频。")
                return
            }
            do {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                    PHPhotoLibrary.shared().performChanges {
                        PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: outputURL)
                    } completionHandler: { success, error in
                        if let error { continuation.resume(throwing: error) }
                        else if success { continuation.resume() }
                        else { continuation.resume(throwing: PhotoSaveError.failed) }
                    }
                }
                saveStatus = .saved
            } catch {
                saveStatus = .failed("保存失败：\(error.localizedDescription)")
            }
        }
    }
}

private enum PhotoSaveError: Error {
    case failed
}
#endif
