# Login / Signup Page

A simple, clean login & signup application with **MongoDB**, **Google OAuth**, **Docker**, and **Jenkins CI/CD**.

![Node.js](https://img.shields.io/badge/Node.js-18+-green)
![MongoDB](https://img.shields.io/badge/MongoDB-7-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## Features

- Email & password signup / login
- Google OAuth 2.0 login
- Session-based authentication (stored in MongoDB)
- Password hashing with bcrypt
- Clean, responsive UI
- Docker & Docker Compose setup
- Jenkins CI/CD pipeline
- Health check endpoint (`/health`)

---

## Quick Start (Local)

### Prerequisites

- **Node.js 18+**
- **MongoDB** running locally (install [MongoDB Compass](https://www.mongodb.com/products/compass) for a GUI)

### Steps

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
#    Edit .env and set your values (MongoDB URI, Google OAuth keys, etc.)

# 3. Start the app
npm start

# 4. Open browser
#    http://localhost:3000
```

---

## Quick Start (Docker)

```bash
# Start everything (app + MongoDB)
docker-compose up -d --build

# Open http://localhost:3000

# Stop
docker-compose down
```

---

## Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create a new project (or select existing)
3. Go to **APIs & Services → Credentials**
4. Click **Create Credentials → OAuth 2.0 Client ID**
5. Set **Authorized redirect URI** to: `http://localhost:3000/auth/google/callback`
6. Copy the **Client ID** and **Client Secret** into your `.env` file:

```
GOOGLE_CLIENT_ID=your-client-id-here
GOOGLE_CLIENT_SECRET=your-client-secret-here
```

> The app works without Google OAuth — the button will just redirect back if credentials aren't configured.

---

## MongoDB Compass

Connect MongoDB Compass to `mongodb://localhost:27017/loginpage` to view your data:

- **users** collection — registered user accounts
- **sessions** collection — active sessions

---

## Jenkins CI/CD Pipeline

The included `Jenkinsfile` defines a pipeline with these stages:

1. **Checkout** — pull code from SCM
2. **Install Dependencies** — `npm ci`
3. **Run Tests** — `npm test`
4. **Build Docker Image** — builds the app image
5. **Deploy** — runs `docker-compose up`
6. **Health Check** — verifies the app is running

### Jenkins Setup

1. Install Jenkins with Docker & Node.js plugins
2. Create a **Pipeline** job
3. Point it to your Git repository
4. Jenkins will automatically use the `Jenkinsfile`

---

## Project Structure

```
loginpage/
├── app.js                # Express app entry point
├── config/
│   └── passport.js       # Passport strategies (local + Google)
├── models/
│   └── User.js           # Mongoose user model
├── routes/
│   └── auth.js           # Auth routes (login, signup, Google, logout)
├── views/
│   ├── login.ejs         # Login page
│   ├── signup.ejs         # Signup page
│   └── dashboard.ejs     # Dashboard (after login)
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker Compose (app + MongoDB)
├── Jenkinsfile           # Jenkins CI/CD pipeline
├── .env                  # Environment variables (local)
├── .env.example          # Environment template
└── package.json
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PORT` | Server port | `3000` |
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017/loginpage` |
| `SESSION_SECRET` | Session encryption key | (required) |
| `GOOGLE_CLIENT_ID` | Google OAuth Client ID | (optional) |
| `GOOGLE_CLIENT_SECRET` | Google OAuth Client Secret | (optional) |
| `GOOGLE_CALLBACK_URL` | Google OAuth callback URL | `http://localhost:3000/auth/google/callback` |

---

## License

ISC
