import Foundation

/// The bundled `BSTStroke.mlpackage`: BST-CG-AP, seq_len 30, 25 merged ShuttleSet classes,
/// pose JnB_bone, the averaged-logit ensemble of the 10 released checkpoints
/// (`research/bst/weights/bst_CG_AP_JnB_bone_merged*.pt`).
///
/// Converted and measured in `research/bst/coreml/` (REPORT.md): fp32, max |logit - PyTorch|
/// 2.4e-6 and 100 % argmax agreement over the 749 match-039 clips on CPU_ONLY, CPU_AND_NE and
/// ALL; fp16 flips 1-5 of those 749 and is not shipped.
public enum StrokeModelResource {
    /// Compiled by Xcode into `BSTStroke.mlmodelc`, else the package itself (SwiftPM keeps
    /// `.mlpackage` verbatim; CoreML compiles it on first load).
    public static var modelURL: URL? {
        Bundle.module.url(forResource: "BSTStroke", withExtension: "mlmodelc")
            ?? Bundle.module.url(forResource: "BSTStroke", withExtension: "mlpackage")
    }
}
