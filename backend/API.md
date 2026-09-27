# REST API

统一响应：`{"code":0,"message":"success","data":...}`。登录后请求头：`Authorization: Bearer <token>`。

## 认证
- `POST /api/auth/login`
  - body: `{"username":"admin","password":"Admin@123"}`

## 看板
- `GET /api/dashboard/summary`

## 企业
- `GET /api/companies?keyword=`
- `GET /api/companies/{id}`
- `POST /api/companies`
- `PUT /api/companies/{id}`
- `DELETE /api/companies/{id}`

## 经营指标
- `GET /api/metrics?companyId=&metricCode=&start=&end=`
- `POST /api/metrics`

## 数据接入
- `POST /api/import/excel` multipart
  - `importType=METRIC|COMPLAINT|COMPETITOR`
  - `file=<xlsx/xls>`
- `GET /api/import/history`

## 风险规则
- `GET /api/risk-rules`
- `POST /api/risk-rules`
- `PUT /api/risk-rules/{id}`
- `DELETE /api/risk-rules/{id}`

操作符：`GT / GE / LT / LE / CHANGE_GT / CHANGE_LT`。

## 风险事件
- `GET /api/risks?companyId=&status=&level=`
- `POST /api/risks/{id}/assign` body: `{"assigneeUserId":4}`
- `POST /api/risks/{id}/handle` body: `{"handleResult":"已联系客户并完成故障处理"}`
- `POST /api/risks/{id}/review` body: `{"reviewComment":"处理有效","approved":true}`
- `POST /api/risks/{id}/close`

状态主流程：`OPEN -> ASSIGNED -> HANDLED -> REVIEWED -> CLOSED`。

## 投诉
- `GET /api/complaints?companyId=&category=`
- `POST /api/complaints`

## 竞品
- `GET /api/competitors?companyId=`
- `POST /api/competitors`

## 知识库 / RAG
- `GET /api/knowledge?companyId=`
- `POST /api/knowledge/upload` multipart
  - `file`
  - `companyId`
  - `deptId`
  - `docType`
  - `securityLevel`
  - `title`
- `GET /api/knowledge/jobs?documentId=` 查询解析/索引任务及进度
- `POST /api/knowledge/{id}/replace` multipart `file` 创建新版本
- `POST /api/knowledge/{id}/retry` 重试失败任务
- `GET /api/knowledge/search?companyId=1&q=续费率下降&topK=5`
- `DELETE /api/knowledge/{id}`

上传接口只负责安全校验、保存原文件和创建任务。后台依次执行解析、质量评分、结构化文本提取、
切片和增量索引；可通过文档的 `ingestStatus` 或任务接口查看结果。失败任务不会丢失，可人工重试。

## AI Agent
- `POST /api/ai/analyze`

```json
{
  "companyId": 1,
  "question": "为什么客户流失风险升高？请给出证据与建议。",
  "topK": 5
}
```

- `GET /api/ai/history?companyId=1`

## 系统权限
- `GET /api/system/users`
- `POST /api/system/users`
- `PUT /api/system/users/{id}/roles`
- `GET /api/system/roles`
- `GET /api/system/departments`
