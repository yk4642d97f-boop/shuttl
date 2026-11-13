# HitNet 手机输入顺序修复

问题：`RallyExtractor.swift` 写出的 `player_box` / `player_foot_world` 是 **far 在前、near 在后**，
Python `common.build_cache` 按原样训练。但手机 `HitStage.rawMatrix` 将 near 放在前面。
原来的集成测试从 raw 反推球员时也使用了相同错误映射，所以测试曾经无法发现语义颠倒。
旧文档“击球输出不依赖顺序”的说法同样错误；该网络没有球员置换不变性。

仅在验证集 033、041 上交换两个球员的框与脚点输入，其他数据和权重完全相同：

| 模型 | 训练中的正确顺序 F1@±2 | 原手机顺序 F1@±2 |
|---|---:|---:|
| 旧三模型集成 | 94.52% | 54.53% |
| 新训练单模型 | 96.26% | 37.32% |

这是输入顺序的离线复现，不能冒充真实手机整场测试。它证明正确接入和新模型训练同样必要。

修复内容：

- `HitStage.rawMatrix` 改为 far → near，与训练提取器一致。
- 同一帧同一侧有多个记录时，固定使用最小 track ID，与训练提取器一致。
- 修正 `HitNetParityTests` 的球员语义映射和旧 README。
- 加入从带明确 far/near 标签的观测开始的 Swift Package 回归测试，避免测试重复实现的错误。

两项回归测试在修改前失败，在修改后全部通过。完整 iOS HitNet 一致性测试 7 项全部通过（iPhone 16 Pro / iOS 18.3 模拟器）；
结果保存在 `ios_hitnet_order.xcresult` / `ios_hitnet_order.log`，窗口 logits 最大差 7.63e-6。
权重没有变更；代码修改前的备份保存在 `code_backups/hit_player_order/`。

训练结果与云端接口也必须遵守 far-first。双打的四名球员不能靠这个两槽位修复解决；
双打仍需要四人身份和击球归属模型。
