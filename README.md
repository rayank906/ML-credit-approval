P2P Credit Card Approval

## Docker

This repository now includes a root-level `Dockerfile` and `docker-compose.yml` for local development.

Build the image:

    docker build -t ml-credit-approval .

Run the app with Docker Compose:

    docker compose up --build

The application will be available on `http://localhost:8000`.

### Compose services

- `web`: FastAPI application
- `db`: PostgreSQL database
- `redis`: Redis job queue

If you want to persist your own configuration, create a `.env` file from `.env.example` and update the values.

## AWS EC2 Deployment

1. Launch an EC2 instance with a supported Linux distribution.
2. Open ports `22` (SSH) and `8000` (HTTP) in the security group.
3. SSH into the instance and install Docker + Docker Compose:

    sudo apt update
    sudo apt install -y docker.io docker-compose-plugin
    sudo usermod -aG docker $USER
    newgrp docker

4. Clone this repository on the EC2 instance:

    git clone https://github.com/rayank906/ML-credit-approval.git
    cd ML-credit-approval

5. Copy `.env.example` to `.env` and set the required environment variables.

6. Start the stack:

    docker compose up --build -d

7. Verify the service:

    docker compose ps

8. Visit `http://<EC2_PUBLIC_IP>:8000`.

### Notes for production

- Use AWS RDS for PostgreSQL and AWS ElastiCache for Redis in production.
- Update `DATABASE_URL` and `REDIS_URL` in `.env` to point to managed services instead of the local containers.
- Keep `JWT_SECRET` and other secrets out of source control.
