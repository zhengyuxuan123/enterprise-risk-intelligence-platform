-- 审计与追溯（2026-09-22 新增）
--
-- 背景：后端早就有 /api/ai/analysis/{id}/trace（决策留痕）、/lineage（数据血缘）、
-- /export（导出）、/actions + /actions/{id}/decide（处置审批队列），
-- 但前端一个方法都没有，等于「能力在、入口不在」。
-- 本文件补上两条权限码，并把它们授予相应角色。
--
-- 手工执行（**必须带 utf8mb4**：Windows 上 mysql 客户端默认按 GBK 读脚本，
-- 中文会报 Incorrect string value）：
--   mysql -uroot -p123456 --default-character-set=utf8mb4 risk_platform < mysql/audit_permissions.sql

-- 1) 权限码。permission_code 上有 UNIQUE，INSERT IGNORE 天然幂等。
INSERT IGNORE INTO sys_permission (permission_name, permission_code, permission_type)
VALUES ('审计追溯', 'ai:audit', 'API'), ('处置审批', 'ai:approve', 'API');

-- 2) 授权：ai:audit 给所有角色（看得见自己发起的分析的全过程）
INSERT IGNORE INTO sys_role_permission (role_id, permission_id)
SELECT r.id, p.id FROM sys_role r JOIN sys_permission p
WHERE p.permission_code = 'ai:audit';

-- 3) 授权：ai:approve 只给管理员 / 管理层 / 审计（一线与分析岗只能提，不能批）
INSERT IGNORE INTO sys_role_permission (role_id, permission_id)
SELECT r.id, p.id FROM sys_role r JOIN sys_permission p
WHERE p.permission_code = 'ai:approve'
  AND r.role_code IN ('ADMIN', 'MANAGER', 'AUDITOR');

-- 校验
-- SELECT r.role_code, p.permission_code
-- FROM sys_role_permission rp
-- JOIN sys_role r ON r.id = rp.role_id
-- JOIN sys_permission p ON p.id = rp.permission_id
-- WHERE p.permission_code IN ('ai:audit','ai:approve')
-- ORDER BY r.role_code, p.permission_code;
