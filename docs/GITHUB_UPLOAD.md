# GitHub 发布指南

本目录是从开发工作区整理出的公开仓库版本。真实 `.env`、构建产物、依赖目录、运行数据库、向量索引、日志和测试临时文件均未包含。

## 1. 发布前配置

根据需要复制示例环境变量文件：

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
Copy-Item pyagent/.env.example pyagent/.env
```

本地 `.env` 不会被 Git 跟踪。不要将真实 API Key 写入 README、截图或提交记录。

## 2. 创建 GitHub 仓库

在 GitHub 新建一个空仓库，例如 `enterprise-risk-intelligence-platform`。不要勾选自动生成 README、`.gitignore` 或 License，避免第一次推送产生冲突。

## 3. 初始化并推送

安装 Git 后，在当前目录执行：

```powershell
git init
git branch -M main
git add .
git status --short
git commit -m "feat: initialize enterprise risk intelligence platform"
git remote add origin https://github.com/<你的用户名>/enterprise-risk-intelligence-platform.git
git push -u origin main
```

推送前必须检查 `git status --short`，确认不存在 `.env`、数据库文件、日志、`target` 或 `node_modules`。

## 4. 推荐仓库设置

- 在 About 中填写 Spring Boot、Vue、FastAPI、LangGraph、RAG、MySQL、Redis、RabbitMQ。
- 根据个人选择添加许可证；不确定时先保持私有仓库。
- 启用 GitHub secret scanning 和 push protection。
- 将部署密钥配置为 GitHub Actions Secrets，不要写入仓库。

