pipeline {
    agent any
    
    stages {
        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('DVC Pull and Status') {
            steps {
                sh 'dvc pull'
                sh 'dvc status'
            }
        }
        
        stage('Run Pipeline') {
            steps {
                sh 'python -m pipelines.pipeline --config configs/config.yaml'
            }
        }
    }
    
    post {
        always {
            echo 'Pipeline execution completed'
        }
        success {
            echo 'Pipeline succeeded!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}

