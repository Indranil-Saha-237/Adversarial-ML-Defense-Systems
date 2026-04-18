# 🛡️ Adversarial-ML-Defense-Systems

A secure authentication system with integrated **Adversarial Machine Learning Defense** mechanisms. This application demonstrates the contrast between baseline and defended ML models under adversarial attacks, combining secure user authentication with real-time ML defense simulation.

![Node.js](https://img.shields.io/badge/Node.js-18+-green)
![MongoDB](https://img.shields.io/badge/MongoDB-7-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-Latest-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ART](https://img.shields.io/badge/ART-Defense%20Toolbox-red)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Quick Start (Local)](#quick-start-local)
- [Quick Start (Docker)](#quick-start-docker)
- [Google OAuth Setup](#google-oauth-setup)
- [ML Defense System](#ml-defense-system)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Jenkins CI/CD Pipeline](#jenkins-cicd-pipeline)
- [Deployment Guide](#deployment-guide)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project combines a **secure login/signup authentication system** with an **Adversarial ML Defense demonstration**. It showcases:

- **Multi-factor Authentication**: Email/password signup and Google OAuth 2.0
- **Session Management**: Encrypted sessions stored in MongoDB
- **ML Defense Layer**: Baseline vs Defended Random Forest models under adversarial attacks
- **Attack Simulation**: Live HopSkipJump adversarial attack generation
- **Production-Ready**: Docker containerization and Jenkins CI/CD integration

The system demonstrates how machine learning models can be hardened against adversarial attacks using Adversarial Robustness Toolbox (ART).

---

## ✨ Features

### Authentication Features
- ✅ **Email & Password Signup/Login** - Local authentication with bcrypt hashing
- ✅ **Google OAuth 2.0** - Third-party social authentication
- ✅ **Session-Based Auth** - Persistent sessions stored in MongoDB
- ✅ **Password Security** - bcryptjs with salt rounds of 10
- ✅ **Flash Messages** - Real-time user feedback on auth events

### ML Defense Features
- ✅ **Baseline Model** - Undefended Random Forest classifier (89.9% vulnerable to attacks)
- ✅ **Defended Model** - Adversarially trained Random Forest (92.6% robust)
- ✅ **Live Attack Simulation** - HopSkipJump evasion attacks on demand
- ✅ **Model Comparison** - Side-by-side accuracy metrics
- ✅ **Defense Metrics** - Real-time defense effectiveness tracking

### Infrastructure Features
- ✅ **Docker & Docker Compose** - Containerized deployment
- ✅ **Jenkins CI/CD Pipeline** - Automated build, test, and deploy
- ✅ **Health Check Endpoints** - Service availability monitoring
- ✅ **CORS Support** - Cross-origin communication between frontend and ML API
- ✅ **Responsive UI** - Clean, modern user interface

---

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Frontend** | Node.js + EJS | 18+ | Server-side templating |
| **Backend** | Express.js | 5.2.1 | REST API & routing |
| **Database** | MongoDB | 7 | User & session storage |
| **Authentication** | Passport.js | 0.7.0 | Auth strategies |
| **ML Engine** | Flask | Latest | REST API for ML models |
| **ML Framework** | Scikit-learn | Latest | Random Forest models |
| **Defense Lib** | ART (Adversarial Robustness Toolbox) | Latest | Adversarial attack simulation |
| **Containerization** | Docker & Docker Compose | Latest | Service orchestration |
| **CI/CD** | Jenkins | Latest | Pipeline automation |
| **Password Hash** | bcryptjs | 3.0.3 | Secure password hashing |
| **Session Store** | connect-mongo | 6.0.0 | MongoDB session storage |
| **HTTP Client** | Axios | 1.13.6 | API communication |
| **Environment** | dotenv | 17.3.1 | Configuration management |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Frontend Layer                            │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Login Page │  │  Signup Page │  │  Dashboard/ML UI   │  │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬──────────┘  │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼──────────────┐
│         │   Express.js App (Port 3000)       │             │
│  ┌──────▼──────────────────────────────────────────┐        │
│  │  Authentication Routes                          │        │
│  │  - Local: login, signup, logout                │        │
│  │  - OAuth: /auth/google, /auth/google/callback │        │
│  │  - Dashboard: /dashboard (protected)           │        │
│  └──────┬───────────────────────────┬─────────────┘        │
│         │                           │                       │
│    ┌────▼────┐          ┌──────────▼──────────┐            │
│    │ MongoDB  │          │  Axios HTTP Client │            │
│    │ (Port    │          │                    │            │
│    │ 27017)   │          └──────────┬─────────┘            │
│    └──────────┘                     │                       │
└─────────────────────────────────────┼───────────────────────┘
                                      │
┌─────────────────────────────────────┼───────────────────────┐
│         Flask ML Engine (Port 5001) │                       │
│  ┌──────────────────────────────────▼────┐                 │
│  │  ML Defense API Endpoints              │                 │
│  │  - GET /health                         │                 │
│  │  - GET /model/metadata                 │                 │
│  │  - GET /model/comparison               │                 │
│  │  - POST /attack/simulate               │                 │
│  └─────────┬────────────────────┬─────────┘                 │
│            │                    │                           │
│    ┌───────▼──────┐    ┌────────▼────────┐                 │
│    │ Baseline RF  │    │ Defended RF     │                 │
│    │ (Vulnerable) │    │ (Hardened)      │                 │
│    └───────┬──────┘    └────────┬────────┘                 │
│            │                    │                           │
│    ┌───────▼──────┐    ┌────────▼────────┐                 │
│    │ HopSkipJump  │    │ ART Framework   │                 │
│    │ Attack Engine│    │ Defense Layer   │                 │
│    └──────────────┘    └─────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

### For Local Development

- **Node.js 18+** - [Download](https://nodejs.org/)
- **MongoDB 7+** - [Download](https://www.mongodb.com/try/download/community)
- **Python 3.9+** - [Download](https://www.python.org/downloads/)
- **Git** - [Download](https://git-scm.com/)
- **Google OAuth Credentials** - [Setup Guide](#google-oauth-setup)

### For Docker Deployment

- **Docker** - [Download](https://www.docker.com/products/docker-desktop)
- **Docker Compose** - [Included with Docker Desktop](https://docs.docker.com/compose/)

### For Jenkins CI/CD

- **Jenkins** with Docker & Node.js plugins installed
- **Windows agent** (or modify Jenkinsfile for Linux)
- **Git plugin** for SCM integration

---

## 🚀 Quick Start (Local)

### Step 1: Clone Repository

```bash
git clone https://github.com/Indranil-Saha-237/Adversarial-ML-Defense-Systems.git
cd Adversarial-ML-Defense-Systems
```

### Step 2: Setup Environment Variables

Copy the example environment file and configure it:

```bash
cp loginpage/.env.example loginpage/.env
```

Edit `loginpage/.env`:

```env
PORT=3000
MONGODB_URI=mongodb://localhost:27017/loginpage
SESSION_SECRET=your-super-secret-key-change-this-in-production

# Google OAuth (optional - leave as-is if not configuring)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_CALLBACK_URL=http://localhost:3000/auth/google/callback
```

### Step 3: Install Node.js Dependencies

```bash
cd loginpage
npm install
cd ..
```

### Step 4: Setup MongoDB Locally

**Option A: Using MongoDB Community Edition**

```bash
# macOS (Homebrew)
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Linux (Ubuntu)
sudo apt-get install -y mongodb
sudo systemctl start mongodb

# Windows
# Download and run MongoDB installer from https://www.mongodb.com/try/download/community
```

**Option B: Using MongoDB Compass (GUI)**

- Download [MongoDB Compass](https://www.mongodb.com/products/compass)
- It includes a local MongoDB instance
- Connect to `mongodb://localhost:27017`

### Step 5: Setup Python ML Environment

```bash
cd ml
python3 -m venv venv

# Activate virtual environment
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

cd ..
```

### Step 6: Start the Application

**Terminal 1 - Node.js App:**

```bash
cd loginpage
npm start
# Server running on http://localhost:3000
```

**Terminal 2 - Flask ML Engine:**

```bash
cd ml
source venv/bin/activate  # or venv\Scripts\activate on Windows
python app.py
# Flask running on http://localhost:5001
```

**Terminal 3 - Verify MongoDB:**

```bash
# Check MongoDB connection
mongosh
> show databases
> use loginpage
> show collections
```

### Step 7: Access the Application

- **Web UI**: http://localhost:3000
- **ML API Health**: http://localhost:5001/health
- **MongoDB Compass**: Connect to `mongodb://localhost:27017/loginpage`

---

## 🐳 Quick Start (Docker)

### Prerequisites

- Docker & Docker Compose installed
- `.env` file configured in the root directory

### Step 1: Configure Environment

```bash
# Copy and configure environment file
cp loginpage/.env.example .env

# Edit .env in root directory with your Google OAuth credentials
```

### Step 2: Build and Start Services

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Check specific service logs
docker-compose logs -f app        # Node.js app
docker-compose logs -f ml-engine  # Flask API
docker-compose logs -f mongo      # MongoDB
```

### Step 3: Verify Deployment

```bash
# Check app health
curl http://localhost:3000/health

# Check ML engine
curl http://localhost:5001/health

# View MongoDB with Compass
# Connect to: mongodb://localhost:27018/loginpage
```

### Step 4: Access Services

| Service | URL | Purpose |
|---------|-----|---------|
| Web App | http://localhost:3000 | Login/Signup/Dashboard |
| ML API | http://localhost:5001 | ML Defense endpoints |
| MongoDB | mongodb://localhost:27018 | Database (Compass) |

### Step 5: Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (warning: deletes data)
docker-compose down -v

# Stop and remove all (images too)
docker-compose down -v --rmi all
```

---

## 🔑 Google OAuth Setup

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Click "Create Project"
3. Enter project name: `Adversarial-ML-Defense`
4. Click "Create"

### Step 2: Enable Google+ API

1. Go to **APIs & Services** → **Library**
2. Search for "Google+ API"
3. Click "Enable"

### Step 3: Create OAuth 2.0 Credentials

1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **OAuth Client ID**
3. Choose **Web Application**
4. Configure Authorized redirect URIs:
   - `http://localhost:3000/auth/google/callback` (Local)
   - `http://your-production-domain.com/auth/google/callback` (Production)
5. Click "Create"

### Step 4: Copy Credentials

Copy the **Client ID** and **Client Secret** into your `.env` file:

```env
GOOGLE_CLIENT_ID=your-copied-client-id-here
GOOGLE_CLIENT_SECRET=your-copied-client-secret-here
GOOGLE_CALLBACK_URL=http://localhost:3000/auth/google/callback
```

### Step 5: Test Google OAuth

1. Start the application: `npm start`
2. Navigate to http://localhost:3000/login
3. Click "Login with Google"
4. You should be redirected to Google login
5. After authentication, you'll be logged in to the app

> **Note:** If Google OAuth is not configured, the button will redirect back to login without error.

---

## 🧠 ML Defense System

### Overview

The ML Defense System demonstrates **Adversarial Machine Learning** concepts by comparing two Random Forest models:

1. **Baseline Model** - Undefended, vulnerable to adversarial attacks
2. **Defended Model** - Hardened using adversarial training techniques

### Models

| Model | Accuracy | Adversarial Accuracy | Defense Type | Status |
|-------|----------|----------------------|--------------|--------|
| **Baseline RF** | 99.99% | 10% | None | 🔴 VULNERABLE |
| **Defended RF** | 99.99% | 92.6% | Adversarial Training | 🟢 SECURED |

### Adversarial Attacks

**HopSkipJump Attack:**
- Evasion-based attack algorithm
- Generates adversarial examples that fool the model
- Baseline model: Easily fooled (10% accuracy on adversarial examples)
- Defended model: Resists attacks (92.6% accuracy on adversarial examples)

### Key Metrics

- **Clean Accuracy**: Performance on normal, unmodified data
- **Adversarial Accuracy**: Performance on adversarially perturbed data
- **Robustness Gap**: Difference between clean and adversarial accuracy
- **Defense Effectiveness**: How well the model resists attacks

### Features

- **69 Input Features**: Classification based on 69-dimensional feature space
- **10-Sample Attack Simulation**: Real-time attack generation
- **Metadata Endpoint**: Retrieve model information and status
- **Comparison Dashboard**: Visual comparison of baseline vs defended models

---

## 📡 API Reference

### Authentication Endpoints

#### 1. Login Page
```http
GET /login
```
- **Description**: Render login page
- **Auth Required**: No
- **Response**: HTML login form

#### 2. Login (Local Authentication)
```http
POST /login
Content-Type: application/x-www-form-urlencoded

email=user@example.com&password=yourpassword
```
- **Description**: Authenticate with email/password
- **Auth Required**: No
- **Success**: Redirect to `/dashboard` (sets session cookie)
- **Failure**: Redirect to `/login` with error message

#### 3. Signup Page
```http
GET /signup
```
- **Description**: Render signup page
- **Auth Required**: No
- **Response**: HTML signup form

#### 4. Signup (User Registration)
```http
POST /signup
Content-Type: application/x-www-form-urlencoded

name=John Doe&email=john@example.com&password=password123&confirmPassword=password123
```
- **Description**: Create new user account
- **Auth Required**: No
- **Validation**:
  - All fields required
  - Password minimum 6 characters
  - Password confirmation must match
  - Email must be unique
- **Success**: Account created, redirect to `/login`
- **Failure**: Redirect to `/signup` with error message

#### 5. Google OAuth Initiation
```http
GET /auth/google
```
- **Description**: Initiate Google OAuth flow
- **Auth Required**: No
- **Response**: Redirect to Google login consent screen
- **Scopes**: `profile`, `email`

#### 6. Google OAuth Callback
```http
GET /auth/google/callback?code=AUTHORIZATION_CODE
```
- **Description**: Google OAuth callback (handled automatically)
- **Auth Required**: No
- **Success**: User authenticated, redirect to `/dashboard`
- **Failure**: Redirect to `/login` with error message

#### 7. Dashboard (Protected Route)
```http
GET /dashboard
```
- **Description**: User dashboard with ML defense metrics
- **Auth Required**: Yes (session cookie)
- **Response**: HTML dashboard with:
  - User profile information
  - ML model metadata
  - Defense system status
- **Failure**: Redirect to `/login` if not authenticated

#### 8. Logout
```http
GET /logout
```
- **Description**: Destroy session and logout user
- **Auth Required**: Yes
- **Response**: Redirect to `/login`
- **Side Effect**: Session deleted from MongoDB

#### 9. Health Check (App)
```http
GET /health
```
- **Description**: Application health status
- **Auth Required**: No
- **Response**: JSON status
```json
{
  "status": "ok"
}
```

---

### ML Defense API Endpoints

#### 1. ML Engine Health Check
```http
GET http://localhost:5001/health
```
- **Description**: Check ML engine and defense layer status
- **Auth Required**: No
- **Response**:
```json
{
  "status": "online",
  "engine": "RandomForest",
  "defense_active": true
}
```

#### 2. Get Model Metadata
```http
GET http://localhost:5001/model/metadata
```
- **Description**: Retrieve model information and performance metrics
- **Auth Required**: No
- **Response (Defended Model)**:
```json
{
  "model_name": "hardened Random Forest",
  "accuracy": 0.9999,
  "adversarial_accuracy": 0.9260,
  "features_count": 69,
  "status": "SECURED",
  "defense_type": "Adversarial Training"
}
```
- **Response (Baseline Model)**:
```json
{
  "model_name": "Baseline Random Forest",
  "accuracy": 0.9999,
  "adversarial_accuracy": 0.10,
  "features_count": 69,
  "status": "VULNERABLE",
  "defense_type": "None"
}
```

#### 3. Compare Models
```http
GET http://localhost:5001/model/comparison
```
- **Description**: Get side-by-side model comparison data
- **Auth Required**: No
- **Response**:
```json
{
  "labels": ["Clean Accuracy", "Adversarial Accuracy"],
  "baseline": [0.9999, 0.10],
  "defended": [0.9999, 0.9260]
}
```

#### 4. Simulate Adversarial Attack
```http
POST http://localhost:5001/attack/simulate
Content-Type: application/json
```
- **Description**: Generate live adversarial examples using HopSkipJump attack
- **Auth Required**: No
- **Parameters**: None (uses random 10 samples from test set)
- **Response**:
```json
{
  "status": "success",
  "samples_tested": 10,
  "baseline_accuracy": 0.1,
  "defended_accuracy": 0.9
}
```
- **Error Response**:
```json
{
  "status": "error",
  "message": "No model files found in /models"
}
```

---

## 💾 Database Schema

### MongoDB Collections

#### Users Collection

```javascript
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "John Doe",
  "email": "john@example.com",
  "password": "$2a$10$...", // bcrypt hashed
  "googleId": "118364022..." (optional),
  "avatar": "https://lh3.googleusercontent.com/...", (optional)
  "createdAt": ISODate("2026-04-18T05:33:08.000Z")
}
```

**Indexes:**
- `_id` (Primary key)
- `email` (Unique)

**Field Descriptions:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `_id` | ObjectId | Yes | MongoDB auto-generated ID |
| `name` | String | Yes | User's full name |
| `email` | String | Yes | User's email (unique) |
| `password` | String | No | Bcrypt hashed password (null if OAuth only) |
| `googleId` | String | No | Google OAuth ID (null if local auth) |
| `avatar` | String | No | Profile picture URL |
| `createdAt` | Date | Yes | Account creation timestamp |

#### Sessions Collection

```javascript
{
  "_id": "session-hash-id",
  "expires": ISODate("2026-04-19T05:33:08.000Z"),
  "session": {
    "cookie": {
      "originalMaxAge": 86400000,
      "expires": "2026-04-19T05:33:08.000Z",
      "secure": false,
      "httpOnly": true,
      "path": "/"
    },
    "passport": {
      "user": "507f1f77bcf86cd799439011"
    }
  }
}
```

**Auto-managed by:** `connect-mongo` (Express session store)
**TTL Index:** Sessions expire after 24 hours

---

## 📁 Project Structure

```
Adversarial-ML-Defense-Systems/
│
├── 📂 loginpage/                    # Node.js Web Application
│   ├── app.js                       # Express app entry point
│   ├── package.json                 # Dependencies & scripts
│   ├── Dockerfile                   # Docker image for web app
│   │
│   ├── 📂 config/
│   │   └── passport.js              # Passport authentication strategies
│   │
│   ├── 📂 models/
│   │   └── User.js                  # Mongoose User model
│   │
│   ├── 📂 routes/
│   │   └── auth.js                  # Authentication routes
│   │
│   ├── 📂 views/                    # EJS templates
│   │   ├── login.ejs                # Login form
│   │   ├── signup.ejs               # Signup form
│   │   └── dashboard.ejs            # User dashboard
│   │
│   └── 📂 public/                   # Static assets (CSS, JS, images)
│
├── 📂 ml/                           # Python ML Engine
│   ├── app.py                       # Flask API entry point
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Docker image for ML engine
│   │
│   ├── 📂 models/                   # Pre-trained ML models
│   │   ├── baseline_rf.pkl          # Baseline Random Forest
│   │   └── defended_rf.pkl          # Defended Random Forest
│   │
│   └── 📂 data/                     # Test datasets
│       ├── X_test.npy               # Test features
│       └── y_test.npy               # Test labels
│
├── docker-compose.yml               # Docker Compose orchestration
├── Jenkinsfile                      # Jenkins CI/CD pipeline
├── README.md                        # This file
├── LICENSE                          # MIT License
├── .gitignore                       # Git ignore rules
├── .env.example                     # Environment template
└── .env                             # Configuration (git-ignored)
```

---

## ⚙️ Environment Variables

### Web Application (.env in loginpage/ or root directory)

```env
# Server Configuration
PORT=3000                                          # Express server port

# Database
MONGODB_URI=mongodb://localhost:27017/loginpage   # MongoDB connection string

# Session Management
SESSION_SECRET=your-super-secret-key-change-this # Session encryption key (required - change in production)

# Google OAuth (optional - app works without this)
GOOGLE_CLIENT_ID=your-google-client-id           # OAuth Client ID
GOOGLE_CLIENT_SECRET=your-google-client-secret   # OAuth Client Secret
GOOGLE_CALLBACK_URL=http://localhost:3000/auth/google/callback

# ML API (Docker only)
ML_API_URL=http://ml-engine:5001                # Flask ML API endpoint
```

### Environment Variable Descriptions

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `PORT` | Number | No | `3000` | Express server port |
| `MONGODB_URI` | String | No | `mongodb://localhost:27017/loginpage` | MongoDB connection string |
| `SESSION_SECRET` | String | Yes | - | Secret key for session encryption (⚠️ Change in production) |
| `GOOGLE_CLIENT_ID` | String | No | - | Google OAuth Client ID |
| `GOOGLE_CLIENT_SECRET` | String | No | - | Google OAuth Client Secret |
| `GOOGLE_CALLBACK_URL` | String | No | `http://localhost:3000/auth/google/callback` | OAuth callback URL |
| `ML_API_URL` | String | No | `http://127.0.0.1:5001` | ML API base URL |

### Example .env Files

**Local Development:**
```env
PORT=3000
MONGODB_URI=mongodb://localhost:27017/loginpage
SESSION_SECRET=local-development-secret-change-in-production
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_CALLBACK_URL=http://localhost:3000/auth/google/callback
```

**Docker Deployment:**
```env
PORT=3000
MONGODB_URI=mongodb://mongo:27017/loginpage
SESSION_SECRET=production-secret-key-generate-random-string
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_CALLBACK_URL=http://your-domain.com/auth/google/callback
ML_API_URL=http://ml-engine:5001
```

**Production with Environment-Specific Config:**
```env
# Production config (hosted on domain)
PORT=3000
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/loginpage
SESSION_SECRET=<generate-secure-random-string>
GOOGLE_CLIENT_ID=<production-client-id>
GOOGLE_CLIENT_SECRET=<production-client-secret>
GOOGLE_CALLBACK_URL=https://your-domain.com/auth/google/callback
ML_API_URL=http://ml-engine:5001
NODE_ENV=production
```

---

## 🔄 Jenkins CI/CD Pipeline

### Overview

The Jenkins pipeline automates building, testing, and deploying the application using Docker.

### Pipeline Stages

#### Stage 1: Checkout
```groovy
stages {
    stage('Checkout') {
        steps {
            checkout scm  // Clones latest code from repository
        }
    }
}
```
- Clones the repository
- Sets up Git branch tracking

#### Stage 2: Setup Environment
```groovy
stage('Setup Environment') {
    steps {
        withCredentials([file(credentialsId: 'Capstone_env_file', variable: 'ENV_FILE')]) {
            bat 'copy %ENV_FILE% .env'  // Copy credentials from Jenkins secure storage
        }
        bat 'cd loginpage && npm install'
    }
}
```
- Retrieves `.env` file from Jenkins credentials (secure storage)
- Installs Node.js dependencies
- Automatically cleans up `.env` after build (see post section)

#### Stage 3: Build & Deploy
```groovy
stage('Build & Deploy') {
    steps {
        bat 'docker-compose down || exit 0'     // Gracefully stop existing containers
        bat 'docker-compose up -d --build'       // Build images and start containers
    }
}
```
- Stops existing Docker containers
- Builds Docker images for all services
- Starts all services in detached mode

#### Stage 4: Health Check
```groovy
stage('Health Check') {
    steps {
        echo 'Waiting 15 seconds for services to stabilize...'
        bat 'powershell -Command "Start-Sleep -Seconds 15"'
        bat 'curl -v -f http://127.0.0.1:3000/health || exit 1'
    }
}
```
- Waits 15 seconds for services to start
- Checks `/health` endpoint to verify app is running
- Fails build if health check fails

### Post-Build Actions

#### Always Execute
```groovy
post {
    always {
        bat 'if exist .env del .env'  // Remove sensitive .env file
    }
}
```
- Deletes `.env` file after build (security best practice)

#### On Success
```groovy
success {
    echo 'Pipeline completed successfully!'
}
```

#### On Failure
```groovy
failure {
    echo 'Pipeline failed! Grabbing logs...'
    bat 'docker-compose logs'  // Display Docker logs for debugging
}
```

### Jenkins Setup Instructions

#### Prerequisites
- Jenkins installed with plugins:
  - Docker
  - Pipeline
  - Git
  - Node.js
- Windows agent (or modify Jenkinsfile for Linux)
- Docker and Docker Compose installed

#### Configuration Steps

1. **Create Jenkins Credentials**
   - Jenkins Dashboard → Manage Credentials
   - Add Credentials → File
   - ID: `Capstone_env_file`
   - Upload your `.env` file

2. **Create Pipeline Job**
   - New Item → Pipeline
   - Name: `Adversarial-ML-Defense-Pipeline`
   - Pipeline script from SCM
   - SCM: Git
   - Repository URL: `https://github.com/Indranil-Saha-237/Adversarial-ML-Defense-Systems.git`
   - Script Path: `Jenkinsfile`

3. **Build Triggers** (Optional)
   - GitHub hook trigger for GITScm polling
   - Poll SCM (e.g., `H/5 * * * *` for every 5 minutes)

4. **Execute Build**
   - Click "Build Now"
   - Monitor pipeline progress in real-time
   - View stage logs for debugging

### Pipeline Execution Flow

```
┌─────────────────┐
│    Checkout     │ Pull latest code
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Setup Environment   │ Copy .env, install dependencies
└────────┬────────────┘
         │
         ▼
┌───────────────────────┐
│  Build & Deploy       │ docker-compose up
└────────┬──────────────┘
         │
         ▼
┌────────────────────┐
│   Health Check     │ Verify endpoints
└────────┬───────────┘
         │
    ┌────┴─────┐
    │           │
    ▼           ▼
┌────────┐  ┌────────┐
│Success │  │Failure │ Show logs
└────────┘  └────────┘
    │           │
    └─────┬─────┘
          ▼
   ┌────────────────┐
   │ Cleanup .env   │ Remove sensitive file
   └────────────────┘
```

---

## 🚀 Deployment Guide

### Local Development Deployment

```bash
# 1. Clone and setup
git clone https://github.com/Indranil-Saha-237/Adversarial-ML-Defense-Systems.git
cd Adversarial-ML-Defense-Systems

# 2. Install dependencies
cd loginpage && npm install && cd ..

# 3. Setup Python environment
cd ml && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cd ..

# 4. Configure environment
cp loginpage/.env.example loginpage/.env
# Edit loginpage/.env with your configuration

# 5. Start services in separate terminals
# Terminal 1:
cd loginpage && npm start

# Terminal 2:
cd ml && source venv/bin/activate && python app.py

# Terminal 3:
mongod  # Or use MongoDB Compass GUI
```

### Docker Deployment (Recommended for Production)

```bash
# 1. Clone repository
git clone https://github.com/Indranil-Saha-237/Adversarial-ML-Defense-Systems.git
cd Adversarial-ML-Defense-Systems

# 2. Configure environment
cp loginpage/.env.example .env
# Edit .env with production settings

# 3. Build and start services
docker-compose up -d --build

# 4. Verify deployment
curl http://localhost:3000/health
curl http://localhost:5001/health

# 5. Monitor logs
docker-compose logs -f

# 6. Stop services
docker-compose down
```

### Production Deployment Checklist

- [ ] Change `SESSION_SECRET` to a random secure string
- [ ] Update `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`
- [ ] Set `GOOGLE_CALLBACK_URL` to production domain
- [ ] Configure MongoDB Atlas connection (MongoDB Cloud)
- [ ] Update `MONGODB_URI` to production database
- [ ] Enable HTTPS/SSL certificates
- [ ] Configure firewall rules
- [ ] Setup monitoring and logging
- [ ] Configure automated backups
- [ ] Test health check endpoints
- [ ] Load test the application
- [ ] Document deployment procedure
- [ ] Setup alerting for failures

---

## 🔒 Security Best Practices

### Authentication Security

✅ **Password Hashing**
- Uses bcryptjs with salt rounds of 10
- Prevents rainbow table attacks
- Passwords never stored in plain text

✅ **Session Management**
- Sessions encrypted and stored in MongoDB
- 24-hour session expiration
- HttpOnly cookies prevent XSS attacks
- Session secrets should be unique per environment

✅ **OAuth 2.0 Implementation**
- Secure redirect URIs configured
- Client secrets never exposed in frontend
- Automatic token refresh handling
- Fallback for missing credentials

### Environment & Secrets

✅ **Environment Variables**
- `.env` file never committed to Git (included in `.gitignore`)
- Use `.env.example` as template
- Rotate secrets regularly
- Different secrets for dev/staging/production

✅ **Jenkins CI/CD Security**
- `.env` file stored in Jenkins credentials (encrypted)
- Automatically deleted after build
- No secrets in Jenkinsfile (visible in Git)
- Build logs sanitized of sensitive data

### Data Security

✅ **Database Protection**
- MongoDB running in containerized environment
- Network isolation in Docker
- Unique email index prevents duplicates
- No sensitive data in logs

✅ **CORS Configuration**
- Restricted to `http://localhost:3000` in development
- Should be updated for production domain
- Prevents cross-site request attacks

### Docker Security

✅ **Container Hardening**
- Alpine base images for smaller attack surface
- Non-root user execution (not implemented, recommended to add)
- Health checks enabled
- Secrets passed via environment variables

✅ **Dependency Management**
- Pin specific versions in package.json
- Regular updates for security patches
- Use `npm audit` to check vulnerabilities
- Implement dependency scanning in CI/CD

### API Security

✅ **Endpoint Protection**
- Protected routes require authentication
- Health checks available without auth
- Rate limiting not implemented (recommended for production)
- Input validation on all forms

### SSL/TLS (For Production)

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Use Let's Encrypt for production
# https://letsencrypt.org/
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: MongoDB Connection Failed

**Symptoms:**
```
MongoDB connection error: connect ECONNREFUSED 127.0.0.1:27017
```

**Solutions:**
- Verify MongoDB is running: `mongosh` or `mongo`
- Check MongoDB port: `netstat -an | grep 27017`
- On macOS: `brew services start mongodb-community`
- On Linux: `sudo systemctl start mongodb`
- Verify MONGODB_URI in `.env` is correct

**Verification:**
```bash
# Connect to MongoDB
mongosh
# Should open MongoDB shell

# List databases
show databases

# Select database
use loginpage

# View collections
show collections
```

---

#### Issue 2: Port Already in Use

**Symptoms:**
```
Error: listen EADDRINUSE: address already in use :::3000
```

**Solutions:**
```bash
# Find process using port 3000
lsof -i :3000  # macOS/Linux
netstat -ano | findstr :3000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or change PORT in .env
PORT=3001
```

---

#### Issue 3: Google OAuth Not Working

**Symptoms:**
- Google button redirects back to login
- "No account with that email" error

**Solutions:**
1. Verify `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env`
2. Check callback URL matches Google Cloud Console configuration
3. Ensure Google+ API is enabled in Google Cloud Console
4. Clear browser cache and cookies
5. Check browser console for JavaScript errors

**Test OAuth without credentials:**
```bash
# This is expected - OAuth will not work
GOOGLE_CLIENT_ID=your-google-client-id
# Should show an error or redirect

# Remove credentials to skip OAuth
# (app works without it)
GOOGLE_CLIENT_ID=your-google-client-id  # Keep original
```

---

#### Issue 4: ML Engine 404 Error

**Symptoms:**
```
Dashboard shows: "modelName": "SYSTEM OFFLINE"
ML API Error: 404 Not Found
```

**Solutions:**
- Verify Flask API is running: `python app.py`
- Check ML_API_URL in Node.js `.env`
- Verify model files exist: `ml/models/baseline_rf.pkl` and `ml/models/defended_rf.pkl`
- Check Flask logs for errors
- Test ML API directly: `curl http://localhost:5001/health`

**Verify Model Files:**
```bash
ls -la ml/models/
# Should show:
# - baseline_rf.pkl
# - defended_rf.pkl
```

---

#### Issue 5: Docker Build Fails

**Symptoms:**
```
ERROR: Service 'app' failed to build: ...
```

**Solutions:**
```bash
# Clean up Docker images and volumes
docker-compose down -v
docker system prune -a

# Rebuild from scratch
docker-compose up -d --build

# View build logs
docker-compose logs app

# Check Docker disk space
docker system df
```

---

#### Issue 6: Jenkins Health Check Fails

**Symptoms:**
```
curl: (7) Failed to connect to 127.0.0.1 port 3000: Connection refused
Health Check Failed
```

**Solutions:**
- Increase wait time in Jenkinsfile: Change `Start-Sleep -Seconds 15` to `30`
- Verify Docker is running on agent
- Check Docker daemon logs
- Verify port 3000 is not blocked by firewall
- Check available system resources (RAM, disk)

---

#### Issue 7: "No Models Found" Error from ML API

**Symptoms:**
```json
{"error": "No model files found in /models"}
```

**Solutions:**
1. Verify Docker volume mapping in `docker-compose.yml`
2. Check model files exist in host system: `ml/models/`
3. Verify Docker volume permissions
4. Recreate volumes: `docker-compose down -v && docker-compose up -d --build`

---

### Debug Mode

**Enable Express Debug Logging:**
```bash
DEBUG=* npm start
```

**Enable MongoDB Debug:**
```bash
mongosh --verbose
```

**View Docker Container Logs:**
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs app
docker-compose logs ml-engine
docker-compose logs mongo

# Follow logs in real-time
docker-compose logs -f app

# Last 100 lines
docker-compose logs --tail=100 app
```

**Access Container Shell:**
```bash
# Node.js app
docker-compose exec app /bin/sh

# Flask app
docker-compose exec ml-engine /bin/bash

# MongoDB
docker-compose exec mongo mongosh
```

---

## 📚 Learning Resources

- **Express.js**: https://expressjs.com/
- **Passport.js**: http://www.passportjs.org/
- **MongoDB**: https://www.mongodb.com/docs/
- **Adversarial Robustness Toolbox**: https://github.com/Trusted-AI/adversarial-robustness-toolbox
- **Docker**: https://docs.docker.com/
- **Jenkins**: https://www.jenkins.io/doc/

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Fork & Clone

```bash
# 1. Fork repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/Adversarial-ML-Defense-Systems.git
cd Adversarial-ML-Defense-Systems

# 3. Add upstream remote
git remote add upstream https://github.com/Indranil-Saha-237/Adversarial-ML-Defense-Systems.git
```

### Create Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### Make Changes

```bash
# Test your changes locally
cd loginpage && npm start  # Terminal 1
cd ml && python app.py     # Terminal 2

# Run tests
npm test
```

### Commit & Push

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add description of feature"
# or
git commit -m "fix: address issue with bug"

# Push to your fork
git push origin feature/your-feature-name
```

### Create Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Add title and description
4. Link related issues (#123)
5. Wait for review and CI checks

### Commit Message Guidelines

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

**Examples:**
```
feat(auth): add email verification
fix(ml): correct adversarial accuracy calculation
docs(readme): update installation steps
```

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👤 Author

**Indranil Saha** - [@Indranil-Saha-237](https://github.com/Indranil-Saha-237)

---

## ⭐ Acknowledgments

- **Adversarial Robustness Toolbox (ART)** - IBM's ART for adversarial ML research
- **Express.js Community** - For excellent web framework
- **Passport.js** - For authentication middleware
- **MongoDB** - For document database
- **Docker** - For containerization platform

---

## 📞 Support & Questions

- 📧 Create an [Issue](https://github.com/Indranil-Saha-237/Adversarial-ML-Defense-Systems/issues)
- 💬 Start a [Discussion](https://github.com/Indranil-Saha-237/Adversarial-ML-Defense-Systems/discussions)
- 📖 Check existing documentation
- 🔍 Search closed issues for solutions

---

## 🎯 Project Status

- ✅ Authentication System (Complete)
- ✅ ML Defense Models (Complete)
- ✅ Docker Deployment (Complete)
- ✅ Jenkins CI/CD (Complete)
- 🔄 Unit Tests (In Progress)
- 📋 Kubernetes Support (Planned)
- 📋 Advanced Attack Algorithms (Planned)
- 📋 Real-time Defense Dashboard (Planned)

---

**Last Updated:** April 18, 2026

**Repository:** [Adversarial-ML-Defense-Systems](https://github.com/Indranil-Saha-237/Adversarial-ML-Defense-Systems)