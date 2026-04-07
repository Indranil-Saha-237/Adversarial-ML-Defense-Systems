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
                // 1. Fetch your .env file from Jenkins credentials
                // Replace 'my-capstone-env' with the ID you create in Jenkins
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
                // Wait for services to start
                bat 'timeout /t 15 /nobreak' 
                bat 'curl -f http://localhost:3000/health || exit 1'
            }
        }
    }

    post {
        always {
            // Clean up the .env file so it's not sitting in the workspace
            bat 'del .env'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
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
