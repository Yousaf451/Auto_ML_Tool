# Deploy on Koyeb

Deploy the backend and frontend as two Docker services from the `main` branch.

## Backend service

- Repository: `Yousaf451/Auto_ML_Tool`
- Dockerfile: `backend/Dockerfile`
- Docker context: `backend`
- Exposed port: `8000`
- Health check path: `/`
- Environment variable: `CORS_ORIGINS=https://<frontend-service>-<app>-<org>.koyeb.app`

After the backend deploys, copy its public Koyeb URL.

## Frontend service

- Repository: `Yousaf451/Auto_ML_Tool`
- Dockerfile: `frontend/Dockerfile`
- Docker context: `frontend`
- Exposed port: `80`
- Environment variable: `BACKEND_URL=<backend-public-koyeb-url>`

The frontend proxies `/api` to the backend, so no frontend build-time API URL is required.

## Dashboard flow

1. Create a Koyeb App and deploy the backend as a Docker service from GitHub.
2. Deploy the frontend as a second Docker service from the same repository.
3. Set `BACKEND_URL` on the frontend to the backend's public URL.
4. Set `CORS_ORIGINS` on the backend to the frontend's public URL.
5. Redeploy both services and open the frontend service URL.