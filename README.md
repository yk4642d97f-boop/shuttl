# Shuttl

**让每一拍，都有数据可循。**

Shuttl 是一款 AI 羽毛球分析软件：融合视觉追踪、时序击球识别与空气阻力物理建模，从单机位比赛录像中提取击球、球路与球员运动信息，并估算三维轨迹、球速和落点，为逐拍复盘提供依据。

**Swift / Core ML · Python / PyTorch · FastAPI / Celery · 单目三维重建**

本仓库公开 iOS 应用、分析后端、核心算法、测试和部分研究脚本。源码基于 **2026-10-11** 的项目快照，公开整理于 **2026-11-13**；当前是供阅读和研究的源码版本，模型权重、比赛视频、训练数据和部分图像资源需要另行准备。

## 技术亮点

### 从单机位画面估算三维飞行

结合球场几何、相机标定、重力和二次空气阻力，使用多初始解优化与数值积分拟合羽毛球飞行。输出三维轨迹、初速度、落点及不确定度；观测不足时限制输出，靠线且误差较大时返回不确定。

代码入口：[Python 三维重建](backend/app/services/tracking/reconstruct3d.py)、[Swift 三维重建](iOS/Packages/ShuttlReconstruct/Sources/ShuttlReconstruct)。

### 从逐帧检测到逐拍理解

分析流水线串联球检测、轨迹关联、场地标定、球员姿态、击球时刻检测、三维重建、回合切分、球路分类和比分推断。HitNet 使用球的运动变化与球员位置等时序特征，预测击球时刻与击球方；现有学习模型的验证范围主要是单打。

代码入口：[分析流水线](backend/app/services/tracking/pipeline_v2.py)、[HitNet](backend/app/services/tracking/hitnet_core.py)、[球路分类](backend/app/services/tracking/bst.py)。

### 手机端与云端分析

iOS 侧采用 Swift / Core ML，后端采用 FastAPI / Celery。应用包含分析进度保存、恢复、温控调度和结果回放；云端路径支持异步分析与结果归档。公开版已将个人签名信息和部署地址替换为开发占位配置。

代码入口：[端上流水线](iOS/Packages/ShuttlLive/Sources/ShuttlLive/LiveStages.swift)、[分析入口](iOS/Sources/Services/OnDevice/OnDeviceAnalysis.swift)、[后端](backend/app)。

## 已记录的实验结果

以下是仓库内 **2026-10-07** 的历史研究结果，未在本次公开整理中重新运行。它们描述候选模型及指定评测条件，不等同于任意录像上的完整 App 准确率，也不构成全球排名。

| 指标 | 原基线 | 新候选 | 评测范围 |
| --- | ---: | ---: | --- |
| 击球检测 F1，容差 ±2 帧 | 92.12% | **94.55%** | 5 场既有单打测试比赛，3,499 次标注击球 |
| 击球检测 F1，容差 ±2 帧 | 94.70% | **96.97%** | 本轮 HitNet 训练和选模之外的 2 场隔离评测 |
| 首拍检出率 | 50.5% | **78.9%** | 上述 5 场既有测试比赛，426 次首拍 |
| 击球时刻与球种联合 F1 | 80.18% | **81.53%** | 同一组 5 场测试，3,320 次具有可用球种标签的事件 |

详见 [HitNet 模型卡](research/cloud_upgrade/MODEL_CARD.md) 和 [击球与球种联测](research/cloud_upgrade/STROKE_CHAIN_REPORT.md)。这些评测基于标注回合范围；连续休息段误报、新球馆、双打、球速和落点的独立真值精度仍需专门验证。模型卡中的候选状态以该实验日期为准。

## 浏览源码

| 目录 | 内容 |
| --- | --- |
| [`iOS/`](iOS/) | Swift 应用、19 个 Swift 模块、界面、分析运行时与测试 |
| [`backend/`](backend/) | API、异步任务、视觉与物理算法、评测工具及测试 |
| [`research/`](research/) | 击球检测、球路分类、Core ML 转换、特征提取和评测脚本 |
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | 开发环境、配置入口和缺失资源说明 |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | 第三方研究、代码与许可入口 |

## 开发与授权

从 [开发说明](docs/GETTING_STARTED.md) 开始。本仓库不是已部署的在线分析服务，也不包含可直接安装的 iPhone 应用。

源码公开供查看与技术交流；项目原创代码尚未指定统一的开源许可证。第三方代码保留其原许可和署名，详见 [授权说明](LICENSE.md)。
