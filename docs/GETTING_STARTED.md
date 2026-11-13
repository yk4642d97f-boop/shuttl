# 开发说明

## 版本与资源

此公开版基于 2026-10-11 的源码快照整理，保留应用、后端、算法与部分研究代码。它不是完整二进制发行版。本次公开整理只进行了源码检查、配置格式校验与敏感信息检查，没有重新完成 iOS 真机、GPU 推理或整套实验验收。

以下资源未随公开源码分发：模型权重、Core ML 模型包、训练与评测数据、比赛录像、生成的实验 JSON、部分图片和测试夹具。各研究报告属于历史实验记录；报告引用的部分产物需要自行准备或重新生成。

## iOS

原工程配置为 Xcode 16.2、Swift 5.10，最低 iOS 17.0。项目入口是 `iOS/Shuttl.xcodeproj`，工程配置源为 `iOS/project.yml`，也可在安装 XcodeGen 后重新生成项目。

准备开发环境时需要：

1. 在 Xcode 中设置自己的开发团队和 Bundle Identifier；公开版使用 `com.example.shuttl`。
2. 根据使用路径准备 TrackNetV3 / WASB、RTMPose、HitNet 和 BST 相关模型。模型入口可参考 `iOS/project.yml`、各 Swift Package 和 `research/` 下的转换脚本。
3. 补充图标、模型与测试夹具等缺失资源；资源目录中的清单不代表二进制文件已包含。
4. 若使用自己的云端，在 `iOS/Resources/Info.plist` 设置 `ShuttlAPIBaseURL` 并启用 `ShuttlCloudAnalysisEnabled`。公开版默认关闭云端分析，地址为 `https://api.example.com` 占位符。

本地推理同样需要正确的模型资源，因此不能把关闭云端视为无需配置即可运行。

## 后端

后端工程要求 Python 3.12+；依赖见 `backend/requirements.txt`。云端模型依赖另见 `backend/requirements-cloud-models.txt` 和 `backend/requirements-cloud-models-cuda.txt`。

配置入口：

- `backend/.env.example`：数据库、Redis、认证和存储配置示例。复制为自己的 `.env` 后设置真实配置；示例值不能直接用于公网部署。
- `backend/app/main.py`：FastAPI 应用。
- `backend/app/worker.py`：Celery 应用。
- `backend/docker-compose.yml`：API、worker、dispatcher、PostgreSQL 和 Redis 的开发编排。
- `backend/app/services/tracking/tasks.py`：推理模型与运行选项入口。

启动 API 并不代表模型已经可用。完整分析还需要模型权重、匹配的运行库与推理配置；请先阅读相关模块和测试，再用自己有权使用的录像验收。

## 研究与评测

`research/hitnet/` 包含时序击球检测训练与评测；`research/cloud_upgrade/` 保留扩充训练和联测相关脚本及模型卡；`research/coreml/` 包含姿态模型转换与一致性检查。

原始私有目录和云端路径已替换为 `/path/to/...` 占位路径。需要按本地数据位置修改；研究代码中的历史路径不能直接当成新环境的安装配置。

应分别报告训练集、验证集、测试集、容差、标注范围与模型版本。单回合或已匹配事件的准确率，不等于完整比赛端到端准确率。
