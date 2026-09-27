# Security Policy

## Sensitive configuration

Do not commit real API keys, database passwords, JWT secrets, private keys, or local `.env` files.

Create local configuration from the supplied examples:

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
Copy-Item pyagent/.env.example pyagent/.env
```

Replace all placeholder values locally. The repository `.gitignore` excludes these files.

## Before publishing

1. Run `git status --short` and inspect every file that will be committed.
2. Search the staged content for API keys and private credentials.
3. Revoke and replace a credential immediately if it was ever committed.
4. Keep production credentials in GitHub Actions secrets or the deployment platform's secret manager.

## Reporting a vulnerability

Do not disclose exploitable vulnerabilities in a public issue. Contact the repository owner privately with reproduction steps and the affected version.

