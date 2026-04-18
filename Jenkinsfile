pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                // Securely copies the .env file from Jenkins credentials to your workspace
                withCredentials([file(credentialsId: 'Capstone_env_file', variable: 'ENV_FILE')]) {
                    bat 'copy %ENV_FILE% .env' 
                }
                bat 'cd loginpage && npm install'
            }
        }

        stage('Build & Deploy') {
            steps {
                bat 'docker-compose down || exit 0'
                bat 'docker-compose up -d --build'
            }
        }

        stage('Health Check') {
            steps {
                echo 'Waiting 15 seconds for services to stabilize...'
                // FIXED: Uses PowerShell to sleep because 'timeout' is not supported in Jenkins
                bat 'powershell -Command "Start-Sleep -Seconds 15"'
                bat 'curl -v -f http://127.0.0.1:3000/health || exit 1'
            }
        }
    }

    post {
        always {
            // Clean up the .env file after the build for security
            bat 'if exist .env del .env'
        }
        
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed! Grabbing logs...'
            // This will show you exactly why the app crashed
            bat 'docker-compose logs' 
        }
    }
}














// pipeline {
//     agent any

//     stages {

//         stage('Checkout') {
//             steps {
//                 checkout scm
//             }
//         }

//         stage('Setup Environment') {
//             steps {
//                 bat 'cd loginpage && npm install'
//             }
//         }

//         stage('Build & Deploy') {
//             steps {
//                 bat 'docker-compose down || exit 0'
//                 bat 'docker-compose up -d --build'
//             }
//         }

//         stage('Health Check') {
//             steps {
//                 bat 'ping 127.0.0.1 -n 11 > nul'
//                 bat 'curl -f http://localhost:3000/health || exit 1'
//             }
//         }
//     }

//     post {
//         success {
//             echo 'Pipeline completed successfully!'
//         }
//         failure {
//             echo 'Pipeline failed!'
//         }
//     }
// }
