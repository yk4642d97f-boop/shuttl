import SwiftUI

// 关于与致谢: the open-source notices the app must carry (MIT for the detection models,
// BSD / Apache for the ported algorithms). Fonts are Apple system fonts, so none is listed. A quiet paper page:
// no colour block, one small seal, hairline-ruled rows. Licence texts ship in Resources/Legal
// (copied from deliverables/…/NOTICE.txt and licenses/) and open in a sheet.

struct Acknowledgement: Identifiable, Hashable {
    let id: String
    let name: String
    let role: String
    let copyright: String
    let licence: String
    /// Files in Resources/Legal, shown one after another in the licence sheet.
    let files: [String]
    let link: String
}

enum Acknowledgements {
    static let sections: [(title: String, note: String?, items: [Acknowledgement])] = [
        ("羽毛球检测模型", nil, [
            Acknowledgement(id: "tracknetv3", name: "TrackNetV3", role: "羽毛球检测网络与预训练权重",
                            copyright: "Copyright (c) 2024 qaz812345", licence: "MIT",
                            files: ["MIT-TrackNetV3"], link: "github.com/qaz812345/TrackNetV3"),
            Acknowledgement(id: "wasb", name: "WASB-SBDT", role: "HRNet 羽毛球检测权重与网络设计",
                            copyright: "Copyright (c) 2023 NTT Communications Corporation", licence: "MIT",
                            files: ["MIT-WASB-SBDT"], link: "github.com/nttcom/WASB-SBDT"),
            Acknowledgement(id: "hrnet", name: "HRNet", role: "WASB 使用的网络模块",
                            copyright: "Copyright (c) Microsoft; (c) 2019 Leo Xiao; (c) 2019 HRNet", licence: "MIT",
                            files: ["MIT-HRNet-deep-high-resolution-net", "MIT-HigherHRNet"],
                            link: "github.com/leoxiaobin/deep-high-resolution-net.pytorch"),
        ]),
        ("图像与数值算法", "移植为 Swift 并修改", [
            Acknowledgement(id: "opencv", name: "OpenCV 4.x", role: "霍夫直线、距离变换、高斯模糊、单应性等",
                            copyright: "Copyright (C) 2000–2025 Intel, Willow Garage, NVIDIA, AMD, OpenCV Foundation 等",
                            licence: "Apache-2.0", files: ["OpenCV-COPYRIGHT", "Apache-2.0", "OpenCV-legacy-BSD-3-Clause-headers"],
                            link: "github.com/opencv/opencv"),
            Acknowledgement(id: "numpy", name: "NumPy", role: "成对求和、排序、中位数与分位数语义",
                            copyright: "Copyright (c) 2005–2023, NumPy Developers", licence: "BSD-3-Clause",
                            files: ["BSD-3-Clause-NumPy"], link: "numpy.org"),
            Acknowledgement(id: "scipy", name: "SciPy", role: "least_squares（trf 方法）",
                            copyright: "Copyright (c) 2001–2002 Enthought, Inc. 2003–2024, SciPy Developers",
                            licence: "BSD-3-Clause", files: ["BSD-3-Clause-SciPy"], link: "scipy.org"),
            Acknowledgement(id: "lapack", name: "LAPACK", role: "特征向量约定、线性方程求解",
                            copyright: "Copyright (c) 1992–2025 University of Tennessee, UC Berkeley, CU Denver",
                            licence: "modified BSD", files: ["LAPACK-modified-BSD"], link: "netlib.org/lapack"),
        ]),
    ]

    /// Research the analysis builds on (citation only: no code or data from these ships).
    static let citations: [String] = [
        "Huang et al. TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications. AVSS 2019.",
        "Sun et al. TrackNetV2: Efficient Shuttlecock Tracking Network. ICPAI 2020.",
        "Chen, Wang. TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification. ACM MM Asia 2023.",
        "Tarashima et al. Widely Applicable Strong Baseline for Sports Ball Detection and Tracking. BMVC 2023.",
        "Shuttlecock Trajectory Dataset · 林育慶、孫念恩、Tsì-Uí İk（阳明交通大学 CoachAI）。以上模型权重由原作者在此数据集上训练。",
    ]

    static func text(of file: String) -> String? {
        guard let url = Bundle.main.url(forResource: file, withExtension: "txt", subdirectory: "Legal") else { return nil }
        return try? String(contentsOf: url, encoding: .utf8)
    }
}

struct AcknowledgementsView: View {
    @State private var open: Acknowledgement?
    @State private var showsFullNotice = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: AppSpacing.lg) {
                header
                ForEach(Array(Acknowledgements.sections.enumerated()), id: \.offset) { i, section in
                    VStack(alignment: .leading, spacing: AppSpacing.sm) {
                        SectionHeader(title: section.title, folio: String(format: "%02d", i + 1))
                        if let note = section.note {
                            Text(note).font(AppFont.caption).foregroundStyle(Color.inkSoft)
                                .padding(.top, -AppSpacing.xs)
                        }
                        VStack(spacing: 0) {
                            ForEach(section.items) { item in
                                Button { open = item } label: { row(item) }
                                    .buttonStyle(ScaleButtonStyle())
                                if item.id != section.items.last?.id { AppDivider() }
                            }
                        }
                    }
                }
                citations
                footer
            }
            .padding(.horizontal, AppSpacing.gutter)
            .padding(.bottom, AppSpacing.xl)
        }
        .tabBarClearance()
        .paperPage()
        #if DEBUG
        .demoScrollAnchor()
        #endif
        .navigationTitle("关于与致谢")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $open) { item in
            LicenceSheet(title: item.name, subtitle: item.licence,
                         text: item.files.compactMap(Acknowledgements.text(of:)).joined(separator: "\n\n\n"))
        }
        .sheet(isPresented: $showsFullNotice) {
            LicenceSheet(title: "完整声明", subtitle: "NOTICE.txt",
                         text: Acknowledgements.text(of: "NOTICE") ?? "")
        }
        #if DEBUG
        .task {
            let args = ProcessInfo.processInfo.arguments
            guard let i = args.firstIndex(of: "-UIDemoOpenLicence"), i + 1 < args.count else { return }
            try? await Task.sleep(for: .milliseconds(700))
            open = Acknowledgements.sections.flatMap(\.items).first { $0.id == args[i + 1] }
        }
        #endif
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: AppSpacing.xs) {
                Text("SHUTTL · 关于").instrumentLabel(AppFont.labelSmall)
                Text("关于与致谢")
                    .font(AppFont.title1)
                    .foregroundStyle(Color.ink)
                    .accessibilityAddTraits(.isHeader)
                Text("Shuttl 用到了下面这些开源作品和公开研究。感谢所有作者。点开任意一项可以看许可证全文。")
                    .font(AppFont.callout)
                    .foregroundStyle(Color.inkSoft)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: AppSpacing.sm)
            Seal(caption: "致谢", symbol: "heart.fill", fill: .butter, size: 64, tilt: 10)
        }
        .padding(.top, AppSpacing.md)
    }

    private func row(_ item: Acknowledgement) -> some View {
        HStack(alignment: .top, spacing: AppSpacing.md) {
            VStack(alignment: .leading, spacing: 3) {
                Text(item.name)
                    .font(AppFont.numeral(19, relativeTo: .headline))
                    .foregroundStyle(Color.ink)
                Text(item.role)
                    .font(AppFont.callout)
                    .foregroundStyle(Color.ink)
                    .fixedSize(horizontal: false, vertical: true)
                Text(item.copyright)
                    .font(AppFont.caption)
                    .foregroundStyle(Color.inkSoft)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: AppSpacing.sm)
            VStack(alignment: .trailing, spacing: 6) {
                Text(item.licence)
                    .instrumentLabel(AppFont.labelSmall, color: .ink)
                    .textCase(nil)
                    .fixedSize()
                Image(systemName: "chevron.right").font(.system(size: 12, weight: .bold)).foregroundStyle(Color.inkFaint)
            }
        }
        .padding(.vertical, AppSpacing.sm + 4)
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityHint("打开 \(item.licence) 许可证全文")
    }

    private var citations: some View {
        VStack(alignment: .leading, spacing: AppSpacing.sm) {
            SectionHeader(title: "学术引用", folio: String(format: "%02d", Acknowledgements.sections.count + 1))
            Text("只引用论文，App 里不包含它们的代码或数据。")
                .font(AppFont.caption)
                .foregroundStyle(Color.inkSoft)
            ForEach(Acknowledgements.citations, id: \.self) { c in
                HStack(alignment: .firstTextBaseline, spacing: AppSpacing.sm) {
                    Text("·").font(AppFont.callout).foregroundStyle(Color.inkSoft)
                    Text(c)
                        .font(AppFont.caption)
                        .foregroundStyle(Color.ink)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: AppSpacing.sm) {
            AppDivider()
            Text("Shuttl 使用了 Apple 的 Vision、Core ML、AVFoundation、MetricKit 等系统框架。iPhone 是 Apple Inc. 的商标。App 图标和其他美术由 Shuttl 团队制作。")
                .font(AppFont.caption)
                .foregroundStyle(Color.inkSoft)
                .fixedSize(horizontal: false, vertical: true)
            Button { showsFullNotice = true } label: {
                Label("查看完整声明（NOTICE）", systemImage: "doc.text")
                    .font(AppFont.callout.weight(.semibold))
                    .foregroundStyle(Color.cobalt)
            }
            .buttonStyle(ScaleButtonStyle())
            Text("Shuttl \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")")
                .instrumentLabel(AppFont.labelSmall)
        }
    }
}

/// A licence's full text on paper. Every word is kept; only the files' hard ~70-column
/// wraps inside ordinary paragraphs are joined, so a paragraph wraps to the phone's width.
/// Structural lines (copyright notices, separators, list items, headings) keep their breaks.
struct LicenceSheet: View {
    let title: String
    let subtitle: String
    let text: String
    @Environment(\.dismiss) private var dismiss

    static func reflow(_ text: String) -> String {
        /// Lines that keep their own line: copyright notices, separators, list items,
        /// indented blocks, and short all-caps headings ("PREAMBLE", "PERMISSION & CONDITIONS").
        func standsAlone(_ line: String) -> Bool {
            let t = line.trimmingCharacters(in: .whitespaces)
            guard let first = t.first else { return true }
            if t.hasPrefix("Copyright") || t.hasPrefix("(c)") || t.hasPrefix("[") { return true }
            if "-=*•·".contains(first) || line.hasPrefix("  ") { return true }
            if first.isNumber, t.dropFirst().first.map({ $0 == "." || $0 == ")" || $0.isNumber }) == true { return true }
            let heading = t.count <= 40 && t == t.uppercased() && t.rangeOfCharacter(from: .letters) != nil
                && !(t.last.map { ".,;:".contains($0) } ?? false)
            return heading
        }
        var out: [String] = []
        var paragraph: [String] = []
        func flush() {
            if !paragraph.isEmpty { out.append(paragraph.joined(separator: " ")); paragraph = [] }
        }
        for raw in text.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n") {
            let line = raw.replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression)
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                flush(); out.append("")
            } else if standsAlone(line) {
                flush(); out.append(line)
            } else {
                paragraph.append(line.trimmingCharacters(in: .whitespaces))
            }
        }
        flush()
        return out.joined(separator: "\n")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(subtitle).instrumentLabel(AppFont.labelSmall).textCase(nil)
                    Text(title).font(AppFont.title2).foregroundStyle(Color.ink)
                }
                Spacer()
                IconButton(systemName: "xmark", size: 36, iconSize: 13, style: .filled, label: "关闭") { dismiss() }
            }
            .padding(AppSpacing.gutter)
            AppDivider()
            ScrollView {
                Text(text.isEmpty ? "（没有找到许可证文件）" : Self.reflow(text))
                    .font(.system(.caption, design: .monospaced))
                    .lineSpacing(2)
                    .foregroundStyle(Color.ink)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(AppSpacing.gutter)
            }
        }
        .paperPage()
        .presentationDragIndicator(.visible)
    }
}
